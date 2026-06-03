// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_engine.hpp"

#include "4C_binstrategy.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_pstream.hpp"
#include "4C_particle_algorithm_utils.hpp"
#include "4C_particle_engine_communication_utils.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_container_bundle.hpp"
#include "4C_particle_engine_object.hpp"
#include "4C_particle_engine_runtime_vtp_writer.hpp"
#include "4C_particle_engine_unique_global_id.hpp"
#include "4C_particle_input.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_TimeMonitor.hpp>

#include <memory>
#include <unordered_map>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::ParticleEngine::ParticleEngine(MPI_Comm comm, const Teuchos::ParameterList& params)
    : comm_(comm),
      myrank_(Core::Communication::my_mpi_rank(comm)),
      params_(params),
      typevectorsize_(0),
      particlecontainerbundle_(std::make_shared<ParticleContainerBundle>()),
      particleuniqueglobalidhandler_(std::make_unique<UniqueGlobalIdHandler>(comm_, "particle")),
      validownedparticles_(false),
      validghostedparticles_(false),
      validparticleneighbors_(false),
      validglobalidtolocalindex_(false),
      validdirectghosting_(false),
      verlet_skin_factor_(params.get<double>("VERLET_SKIN_FACTOR"))
{
  // init binning strategy
  init_binning_strategy();

  // init particle runtime vtp writer
  init_particle_vtp_writer();
}

Particle::ParticleEngine::~ParticleEngine() = default;

void Particle::ParticleEngine::setup(
    const std::map<ParticleType, std::set<ParticleState>>& particlestatestotypes)
{
  // setup binning strategy
  setup_binning_strategy();

  // setup particle container bundle
  setup_particle_container_bundle(particlestatestotypes);

  // setup data storage
  setup_data_storage(particlestatestotypes);

  // setup particle runtime vtp writer
  setup_particle_vtp_writer();

  // setup particle type weights for dynamic load balancing
  setup_type_weights();
}

void Particle::ParticleEngine::write_restart(const int step, const double time) const
{
  // get bin discretization writer
  std::shared_ptr<Core::IO::DiscretizationWriter> binwriter =
      binning_->binstrategy_->bin_discret()->writer();

  binwriter->new_step(step, time);

  // pack particles of all containers
  std::shared_ptr<std::vector<char>> particlebuffer = std::make_shared<std::vector<char>>();
  particlecontainerbundle_->get_packed_particle_objects_of_all_containers(*particlebuffer);

  // write particle data
  binwriter->write_char_data("ParticleData", *Teuchos::rcp(particlebuffer));

  // write restart of unique global identifier handler
  particleuniqueglobalidhandler_->write_restart(binwriter);
}

void Particle::ParticleEngine::read_restart(
    const std::shared_ptr<Core::IO::DiscretizationReader> reader,
    std::vector<ParticleObjShrdPtr>& particlestoread) const
{
  // read particle data
  std::shared_ptr<std::vector<char>> particledata = std::make_shared<std::vector<char>>();
  reader->read_char_vector(particledata, "ParticleData");

  Core::Communication::UnpackBuffer buffer(*particledata);
  while (!buffer.at_end())
  {
    std::shared_ptr<Core::Communication::ParObject> object(Core::Communication::factory(buffer));
    ParticleObjShrdPtr particleobject = std::dynamic_pointer_cast<ParticleObject>(object);
    if (particleobject == nullptr) FOUR_C_THROW("received object is not a particle object!");

    // store read particle
    particlestoread.push_back(particleobject);
  }

  // read restart of unique global identifier handler
  particleuniqueglobalidhandler_->read_restart(reader);

  // read restart of runtime vtp writer
  particlevtpwriter_->read_restart(reader);
}

void Particle::ParticleEngine::write_particle_runtime_output(
    const int step, const double time) const
{
  particlevtpwriter_->set_particle_positions_and_states();
  particlevtpwriter_->write_to_disk(time, step);
}

void Particle::ParticleEngine::free_unique_global_ids(std::vector<int>& freeuniquegids)
{
  // insert freed global id
  for (const int currfreegid : freeuniquegids)
    particleuniqueglobalidhandler_->insert_freed_global_id(currfreegid);

  // clear after all unique global ids are freed
  freeuniquegids.clear();
}

void Particle::ParticleEngine::get_unique_global_ids_for_all_particles(
    std::vector<ParticleObjShrdPtr>& particlestogetuniquegids)
{
  std::vector<int> requesteduniqueglobalids;
  requesteduniqueglobalids.reserve(particlestogetuniquegids.size());

  // draw requested number of global ids
  particleuniqueglobalidhandler_->draw_requested_number_of_global_ids(requesteduniqueglobalids);

  // number of particles to get unique ids
  int numparticles = particlestogetuniquegids.size();

  // iterate over particles objects
  for (int i = 0; i < numparticles; ++i)
    particlestogetuniquegids[i]->set_particle_global_id(requesteduniqueglobalids[i]);
}

void Particle::ParticleEngine::check_number_of_unique_global_ids()
{
  // get number of particles on all processors
  int numberofparticles = get_number_of_particles();
  MPI_Allreduce(MPI_IN_PLACE, &numberofparticles, 1, MPI_INT, MPI_SUM, comm_);

  // get number of reusable global ids on all processors
  int numberofreusableglobalids =
      particleuniqueglobalidhandler_->get_number_of_reusable_global_ids();
  MPI_Allreduce(MPI_IN_PLACE, &numberofreusableglobalids, 1, MPI_INT, MPI_SUM, comm_);

  // maximum global id
  const int maxglobalid = particleuniqueglobalidhandler_->get_max_global_id();

  // safety check
  if (numberofparticles + numberofreusableglobalids != (maxglobalid + 1))
    FOUR_C_THROW("sum of particles and reusable global ids unequal total global ids: {} + {} != {}",
        numberofparticles, numberofreusableglobalids, (maxglobalid + 1));
}

void Particle::ParticleEngine::erase_particles_outside_bounding_box(
    std::vector<ParticleObjShrdPtr>& particlestocheck)
{
  // get bounding box dimensions
  Core::LinAlg::Matrix<3, 2> boundingbox =
      binning_->binstrategy_->domain_bounding_box_corner_positions();

  // set of particles located outside bounding box
  std::set<int> particlesoutsideboundingbox;

  // number of particles to check
  int numparticles = particlestocheck.size();

  // iterate over particles objects
  for (int i = 0; i < numparticles; ++i)
  {
    // get states of particle
    const ParticleStates& states = particlestocheck[i]->return_particle_states();

    // get position of particle
    const std::vector<double>& pos = states[Position];

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // get type of particles
    ParticleType type = particlestocheck[i]->return_particle_type();

    // get container of owned particles of current particle type
    ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

    if (static_cast<int>(pos.size()) != container->get_state_dim(Position))
      FOUR_C_THROW("dimension of particle state '{}' not valid!", enum_to_state_name(Position));
#endif

    // check particle location with respect to bounding box in each spatial directions
    for (int dim = 0; dim < 3; ++dim)
    {
      // particle located outside bounding box
      if ((pos[dim] < boundingbox(dim, 0)) or (pos[dim] > boundingbox(dim, 1)))
      {
        // insert particle into set
        particlesoutsideboundingbox.insert(i);

#ifdef FOUR_C_ENABLE_ASSERTIONS
        if (particlestocheck[i]->return_particle_global_id() < 0)
          FOUR_C_THROW("no global id assigned to particle!");
#endif

        // insert freed global id
        particleuniqueglobalidhandler_->insert_freed_global_id(
            particlestocheck[i]->return_particle_global_id());

        break;
      }
    }
  }

  // number of particles located outside bounding box
  const int numparticlesoutside = particlesoutsideboundingbox.size();

  // no particles located outside of bounding box
  if (numparticlesoutside == 0) return;

  // put particles to be erased at the end of the vector
  int swapposition = numparticles - 1;

  // iterate in reversed order over particles to be erased
  std::set<int>::reverse_iterator rit;
  for (rit = particlesoutsideboundingbox.rbegin(); rit != particlesoutsideboundingbox.rend(); ++rit)
  {
    // swap position of particle to be erase
    std::swap(particlestocheck[*rit], particlestocheck[swapposition]);

    // set new swap position
    --swapposition;
  }

  // erase particles located outside bounding box
  particlestocheck.resize(numparticles - numparticlesoutside);

  // short screen output
  if (numparticlesoutside)
    Core::IO::cout << "on processor " << myrank_ << " removed " << numparticlesoutside
                   << " particle(s) being outside the computational domain!" << Core::IO::endl;
}

void Particle::ParticleEngine::distribute_particles(
    std::vector<ParticleObjShrdPtr>& particlestodistribute)
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::ParticleEngine::DistributeParticles");

  std::vector<std::vector<ParticleObjShrdPtr>> particlestosend(
      Core::Communication::num_mpi_ranks(comm_));
  std::vector<std::vector<std::pair<int, ParticleObjShrdPtr>>> particlestoinsert(typevectorsize_);

  // determine particles that need to be distributed
  determine_particles_to_be_distributed(particlestodistribute, particlestosend, particlestoinsert);

  // communicate particles
  communicate_particles(particlestosend, particlestoinsert);

  // insert owned particles received from other processors
  insert_owned_particles(particlestoinsert);

  // store particle positions after transfer of particles
  store_positions_after_particle_transfer();

  // relate owned particles to bins
  relate_owned_particles_to_bins();
}

void Particle::ParticleEngine::transfer_particles()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::ParticleEngine::TransferParticles");

  std::vector<std::set<int>> particlestoremove(typevectorsize_);
  std::vector<std::vector<ParticleObjShrdPtr>> particlestosend(
      Core::Communication::num_mpi_ranks(comm_));
  std::vector<std::vector<std::pair<int, ParticleObjShrdPtr>>> particlestoinsert(typevectorsize_);

  // relate owned particles to bins
  if (not validownedparticles_) relate_owned_particles_to_bins();

  // check particles for periodic boundaries/leaving domain
  check_particles_at_boundaries(particlestoremove);

  // determine particles that need to be transferred
  determine_particles_to_be_transferred(particlestoremove, particlestosend);

  // remove particles from containers
  remove_particles_from_containers(particlestoremove);

  // communicate particles
  communicate_particles(particlestosend, particlestoinsert);

  // insert owned particles received from other processors
  insert_owned_particles(particlestoinsert);

  // store particle positions after transfer of particles
  store_positions_after_particle_transfer();

  // relate owned particles to bins
  relate_owned_particles_to_bins();

  // check and decrease the size of all containers of owned particles
  particlecontainerbundle_->check_and_decrease_size_all_containers_of_specific_status(Owned);
}

void Particle::ParticleEngine::ghost_particles()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::ParticleEngine::GhostParticles");

  std::map<int, std::vector<char>> sdata;
  std::map<int, std::map<ParticleType, std::map<int, std::pair<int, int>>>> directghosting;

  // clear all containers of ghosted particles
  particlecontainerbundle_->clear_all_containers_of_specific_status(Ghosted);

  // pack particles to be ghosted directly into send buffers (bypasses ParticleObject)
  pack_particles_to_be_ghosted(sdata);

  // communicate ghost data and insert into containers using cached comm graph (no Allreduce)
  communicate_and_insert_ghost_particles(sdata, directghosting);

  // communicate and build map for direct ghosting
  communicate_direct_ghosting_map(directghosting);

  // check and decrease the size of all containers of ghosted particles
  particlecontainerbundle_->check_and_decrease_size_all_containers_of_specific_status(Ghosted);
}

void Particle::ParticleEngine::refresh_particles() const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::ParticleEngine::RefreshParticles");

  // pack particles to be refreshed directly into send buffers
  std::map<int, std::vector<char>> sdata;
  pack_particles_to_be_refreshed(sdata);

  // R2: look up cached recv sizes for the all-states refresh pattern; on miss compute
  // them locally (no MPI round-trip) from the per-(sender,type) ghost counts recorded
  // during the last ghost rebuild and the deterministic per-particle pack size.
  if (refresh_all_states_recv_sizes_cache_.empty())
    refresh_all_states_recv_sizes_cache_ = compute_refresh_all_states_recv_sizes();

  // communicate and unpack refreshed particles directly into containers
  communicate_refreshed_particles(sdata, &refresh_all_states_recv_sizes_cache_, nullptr);
}

void Particle::ParticleEngine::refresh_particles_of_specific_states_and_types(
    const StatesOfTypesToRefresh& particlestatestotypes) const
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "Particle::ParticleEngine::refresh_particles_of_specific_states_and_types");

  // pack specific states of particles directly into send buffers
  std::map<int, std::vector<char>> sdata;
  pack_specific_states_of_particles_to_be_refreshed(particlestatestotypes, sdata);

  // look up cached recv sizes for this state combination; on miss, compute them locally
  // (no MPI round-trip) from the per-(sender,type) ghost counts recorded during the last
  // ghost rebuild and the deterministic per-particle pack size of the requested pattern.
  auto it = refresh_specific_recv_sizes_cache_.find(particlestatestotypes);
  if (it == refresh_specific_recv_sizes_cache_.end())
  {
    it = refresh_specific_recv_sizes_cache_
             .emplace(
                 particlestatestotypes, compute_refresh_specific_recv_sizes(particlestatestotypes))
             .first;
  }
  communicate_refreshed_particles(sdata, &it->second, nullptr);
}

void Particle::ParticleEngine::dynamic_load_balancing()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::ParticleEngine::dynamic_load_balancing");

  // determine bin weights needed for repartitioning
  determine_bin_weights();

  // distribute bins via recursive coordinate bisection
  binning_->binstrategy_->distribute_bins_recurs_coord_bisection(
      binning_->binrowmap_, binning_->bincenters_, binning_->binweights_);

  // export elements to new layout
  binning_->binstrategy_->bin_discret()->export_row_elements(*binning_->binrowmap_);

  // setup ghosting of bins
  setup_bin_ghosting();

  // determine bin distribution dependent maps/sets
  determine_bin_dis_dependent_maps_and_sets();

  // determine ghosting dependent maps/sets for communication
  determine_ghosting_dependent_maps_and_sets();

  // prepare storage for particle objects
  std::vector<ParticleObjShrdPtr> particlestodistribute;
  particlestodistribute.reserve(get_number_of_particles());

  // get vector of particle objects of all containers
  particlecontainerbundle_->get_vector_of_particle_objects_of_all_containers(particlestodistribute);

  // clear all containers of owned particles
  particlecontainerbundle_->clear_all_containers_of_specific_status(Owned);

  // invalidate particle safety flags
  invalidate_particle_safety_flags();

  // invalidate flag denoting valid relation of half surrounding neighboring bins to owned bins
  binning_->validhalfneighboringbins_ = false;

  // distribute particles to owning processor
  distribute_particles(particlestodistribute);

  // check and decrease the size of all containers of owned particles
  particlecontainerbundle_->check_and_decrease_size_all_containers_of_specific_status(Owned);
}

void Particle::ParticleEngine::hand_over_particles_to_be_removed(
    std::vector<std::set<int>>& particlestoremove)
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::ParticleEngine::hand_over_particles_to_be_removed");

  // number of particles to remove
  int numparticlestoremove = 0;
  for (auto typeIt : particlestoremove) numparticlestoremove += typeIt.size();

  if (numparticlestoremove)
  {
    // remove particles from containers
    remove_particles_from_containers(particlestoremove);

    // check and decrease the size of all containers of owned particles
    particlecontainerbundle_->check_and_decrease_size_all_containers_of_specific_status(Owned);
  }
}

void Particle::ParticleEngine::hand_over_particles_to_be_inserted(
    std::vector<std::vector<std::pair<int, ParticleObjShrdPtr>>>& particlestoinsert)
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::ParticleEngine::hand_over_particles_to_be_inserted");

  // number of particles to insert
  int numparticlestoinsert = 0;
  for (auto typeIt : particlestoinsert) numparticlestoinsert += typeIt.size();

  if (numparticlestoinsert)
  {
    // insert owned particles
    insert_owned_particles(particlestoinsert);
  }
}

void Particle::ParticleEngine::build_particle_to_particle_neighbors()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::ParticleEngine::build_particle_to_particle_neighbors");

  // safety check
  if ((not validownedparticles_) or (not validghostedparticles_))
    FOUR_C_THROW("invalid relation of particles to bins!");

  // relate half neighboring bins to owned bins
  if (not binning_->validhalfneighboringbins_) relate_half_neighboring_bins_to_owned_bins();

  // clear potential particle neighbors
  potentialparticleneighbors_.clear();

  // invalidate flag denoting validity of particle neighbors map
  validparticleneighbors_ = false;

  // initialize per row-bin neighbor-search cost accumulator (R5 load-balance input).
  // Reset to zero for every row bin so empty/skipped bins record a zero cost.
  const int nrowbins_local = binning_->binrowmap_->num_my_elements();
  last_search_cost_per_rowbin_.assign(nrowbins_local, 0.0);

  // loop over row bins
  for (int rowlidofbin = 0; rowlidofbin < nrowbins_local; ++rowlidofbin)
  {
    // get global id of bin
    const int gidofbin = binning_->binrowmap_->gid(rowlidofbin);

    // get local id of bin
    const int collidofbin = binning_->bincolmap_->lid(gidofbin);

    // check if current bin contains owned particles
    if (particlestobins_[collidofbin].empty()) continue;

    // record neighbor-search cost for this bin: the inner candidate-pair loop runs
    // (#owned particles in this bin) x (sum of #particles in all half-neighbor bins)
    // times. Each iteration performs a constant-cost distance check, so this count is
    // a direct proxy for build_particle_to_particle_neighbors' cost contribution from
    // this bin. Stored for consumption by determine_bin_weights at the next LB step.
    {
      std::size_t neighbor_particle_total = 0;
      for (int gidofneighborbin : binning_->halfneighboringbinstobins_[rowlidofbin])
      {
        const int collidofneighbor = binning_->bincolmap_->lid(gidofneighborbin);
        neighbor_particle_total += particlestobins_[collidofneighbor].size();
      }
      last_search_cost_per_rowbin_[rowlidofbin] =
          static_cast<double>(particlestobins_[collidofbin].size()) *
          static_cast<double>(neighbor_particle_total);
    }

    // iterate over owned particles in current bin
    for (const auto& particleIt : particlestobins_[collidofbin])
    {
      // get type of particle
      ParticleType type = particleIt.first;

      // get local index of owned particle
      const int ownedindex = particleIt.second;

      // get container of owned particles of current particle type
      ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

      // get global id of particle
      const int* currglobalid = container->get_ptr_to_global_id(ownedindex);

      // get position of particle
      const double* currpos = container->get_ptr_to_state(Position, ownedindex);

      // iterate over neighboring bins (including current bin)
      for (int gidofneighborbin : binning_->halfneighboringbinstobins_[rowlidofbin])
      {
        // get local id of neighboring bin
        const int collidofneighboringbin = binning_->bincolmap_->lid(gidofneighborbin);

        // check if current neighboring bin contains particles
        if (particlestobins_[collidofneighboringbin].empty()) continue;

        // get status of neighboring particles
        ParticleStatus neighborstatus =
            (binning_->binrowmap_->lid(gidofneighborbin) < 0) ? Ghosted : Owned;

        // iterate over particles in current neighboring bin
        for (const auto& neighborParticleIt : particlestobins_[collidofneighboringbin])
        {
          // get type of neighboring particle
          ParticleType neighbortype = neighborParticleIt.first;

          // get local index of neighboring particle
          const int neighborindex = neighborParticleIt.second;

          // get container of neighboring particle of current particle type
          ParticleContainer* neighborcontainer =
              particlecontainerbundle_->get_specific_container(neighbortype, neighborstatus);

          // get global id of neighboring particle
          const int* neighborglobalid = neighborcontainer->get_ptr_to_global_id(neighborindex);

          // avoid duplicate neighbor pairs and self-neighboring
          if (gidofbin == gidofneighborbin and neighborglobalid[0] <= currglobalid[0]) continue;

          // get position of neighboring particle
          const double* neighborpos = neighborcontainer->get_ptr_to_state(Position, neighborindex);

          // distance vector from owned particle to neighboring particle
          double dist[3];

          // distance between particles considering periodic boundaries
          distance_between_particles(currpos, neighborpos, dist);

          // distance between particles larger than configured pair-search radius (Verlet)
          // or, when Verlet reuse is disabled, larger than the minimum bin size (legacy).
          const double threshold_sq = verlet_active_
              ? (pair_search_radius_ * pair_search_radius_)
              : (binning_->minbinsize_ * binning_->minbinsize_);
          if (dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2] > threshold_sq)
            continue;

          // append potential particle neighbor pair
          potentialparticleneighbors_.push_back(
              std::make_pair(std::make_tuple(type, Owned, ownedindex),
                  std::make_tuple(neighbortype, neighborstatus, neighborindex)));
        }
      }
    }
  }

  // validate flag denoting validity of particle neighbors map
  validparticleneighbors_ = true;

  // snapshot positions for Verlet skin tracking (Light gate reference)
  if (verlet_active_) store_positions_after_neighbor_list_build();
}

void Particle::ParticleEngine::rebuild_potential_neighbors_only()
{
  // Verlet Light-gate path: rebuild the neighbor list locally without redoing
  // ghosting / global-id map / load balance. build_particle_to_particle_neighbors
  // is already a purely local operation that consumes the (still valid) ghost
  // positions refreshed each step by refresh_particles().
  build_particle_to_particle_neighbors();
}

void Particle::ParticleEngine::configure_verlet_pair_search_radius(
    const double max_interaction_distance)
{
  const double requested = max_interaction_distance * (1.0 + verlet_skin_factor_);
  const double minbin = binning_->minbinsize_;

  // pair_search_radius must not exceed min_bin_size: the 3x3x3 bin scan is only
  // complete out to distance min_bin_size, so accepting pairs beyond would risk
  // missing equally-distant pairs sitting in a bin outside the scan window.
  pair_search_radius_ = std::min(requested, minbin);

  // Verlet reuse is meaningful only when there is a real skin budget separate from
  // the existing bin-margin used by the Heavy gate. If the radius was clamped to
  // min_bin_size, the effective skin equals the Heavy-gate margin and Light-gate
  // rebuilds would fire together with the Heavy gate (no benefit). In that case
  // we keep the legacy filter at min_bin_size to remain bit-equivalent.
  const bool real_skin_budget = (verlet_skin_factor_ > 0.0) && (requested < minbin);
  verlet_active_ = real_skin_budget;
  verlet_skin_ = verlet_active_ ? (pair_search_radius_ - max_interaction_distance) : 0.0;

  if (myrank_ == 0)
  {
    if (verlet_active_)
    {
      Core::IO::cout << "Verlet neighbor-list reuse active: pair_search_radius="
                     << pair_search_radius_ << " (skin=" << verlet_skin_
                     << ", min_bin_size=" << minbin
                     << ", R_max=" << max_interaction_distance << ")" << Core::IO::endl;
    }
    else if (verlet_skin_factor_ > 0.0)
    {
      Core::IO::cout << "Verlet neighbor-list reuse requested but bins are too small "
                        "(min_bin_size="
                     << minbin << " <= R_max*(1+skin)=" << requested
                     << "); falling back to legacy filter at min_bin_size." << Core::IO::endl;
    }
  }
}

void Particle::ParticleEngine::store_positions_after_neighbor_list_build()
{
  for (const auto& type : particlecontainerbundle_->get_particle_types())
  {
    ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);
    const int particlestored = container->particles_stored();
    if (particlestored == 0) continue;

    const double* pos = container->get_ptr_to_state(Position, 0);
    double* lastbuildpos = container->get_ptr_to_state(LastNeighborListBuildPosition, 0);
    const int statedim = container->get_state_dim(Position);
    for (int i = 0; i < (statedim * particlestored); ++i) lastbuildpos[i] = pos[i];
  }
}

void Particle::ParticleEngine::build_global_id_to_local_index_map()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::ParticleEngine::build_global_id_to_local_index_map");

  // clear map
  globalidtolocalindex_.clear();

  // invalidate flag denoting validity of map relating particle global ids to local index
  validglobalidtolocalindex_ = false;

  // iterate over particle types
  for (const auto& type : particlecontainerbundle_->get_particle_types())
  {
    // iterate over particle statuses
    for (const auto& status : {Owned, Ghosted})
    {
      // get container of current particle type and current status
      ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, status);

      // get number of particles stored in container
      const int particlestored = container->particles_stored();

      // no particles of current type and current status
      if (particlestored <= 0) continue;

      // get pointer to global id of particles
      int* globalids = container->get_ptr_to_global_id(0);

      // loop over particles in container
      for (int index = 0; index < particlestored; ++index)
      {
        // get global id of particle
        int globalid = globalids[index];

        // add entry to map
        globalidtolocalindex_[globalid] = std::make_shared<LocalIndexTuple>(type, status, index);
      }
    }
  }

  // validate flag denoting validity of map relating particle global ids to local index
  validglobalidtolocalindex_ = true;
}

bool Particle::ParticleEngine::have_valid_particle_connectivity() const
{
  int localcheck = ((validownedparticles_ and validghostedparticles_ and
                     validglobalidtolocalindex_ and validdirectghosting_));

  // check among all processors
  int globalcheck = 0;
  globalcheck = Core::Communication::min_all(localcheck, comm_);

  return globalcheck;
}

bool Particle::ParticleEngine::have_valid_particle_neighbors() const
{
  int localcheck = validparticleneighbors_;

  // check among all processors
  int globalcheck = 0;
  globalcheck = Core::Communication::min_all(localcheck, comm_);

  return globalcheck;
}

const Particle::ParticlesToBins& Particle::ParticleEngine::get_particles_to_bins() const
{
  // safety check
  if ((not validownedparticles_) or (not validghostedparticles_))
    FOUR_C_THROW("invalid map relating particles to bins!");

  return particlestobins_;
}

const Particle::PotentialParticleNeighbors&
Particle::ParticleEngine::get_potential_particle_neighbors() const
{
  // safety check
  if (not validparticleneighbors_) FOUR_C_THROW("invalid particle neighbors!");

  return potentialparticleneighbors_;
}

Particle::LocalIndexTupleShrdPtr Particle::ParticleEngine::get_local_index_in_specific_container(
    int globalid) const
{
  // safety check
  if (not validglobalidtolocalindex_) FOUR_C_THROW("invalid global id to local index map!");

  // get local index of particle in specific container
  auto globalidIt = globalidtolocalindex_.find(globalid);
  if (globalidIt == globalidtolocalindex_.end()) return nullptr;

  return globalidIt->second;
}

std::shared_ptr<Core::IO::DiscretizationWriter>
Particle::ParticleEngine::get_bin_discretization_writer() const
{
  return binning_->binstrategy_->bin_discret()->writer();
}

void Particle::ParticleEngine::relate_all_particles_to_all_procs(
    std::vector<int>& particlestoproc) const
{
  // global ids on this processor
  std::vector<int> thisprocglobalids;

  // iterate over particle types
  for (const auto& type : particlecontainerbundle_->get_particle_types())
  {
    // get container of owned particles of current particle type
    ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

    // get number of particles stored in container
    const int particlestored = container->particles_stored();

    // no particles of current type and current status
    if (particlestored <= 0) continue;

    // get pointer to global id of particles
    int* globalids = container->get_ptr_to_global_id(0);

    // insert global id of particles
    thisprocglobalids.insert(thisprocglobalids.end(), globalids, globalids + particlestored);
  }

  // get maximum global id on this processor
  int thisprocmaxglobalid = 0;
  if (not thisprocglobalids.empty())
    thisprocmaxglobalid = *std::max_element(thisprocglobalids.begin(), thisprocglobalids.end());

  // get maximum global id on all processors
  int allprocmaxglobalid(0);
  allprocmaxglobalid = Core::Communication::max_all(thisprocmaxglobalid, comm_);

  // resize to hold all particles
  const int vecsize = allprocmaxglobalid + 1;
  particlestoproc.resize(vecsize, -1);

  // relate this processor id to its global ids
  for (int globalid : thisprocglobalids) particlestoproc[globalid] = myrank_;

  // communicate global ids between all processors
  MPI_Allreduce(MPI_IN_PLACE, particlestoproc.data(), vecsize, MPI_INT, MPI_MAX, comm_);
}

void Particle::ParticleEngine::get_particles_within_radius(const double* position,
    const double radius, std::vector<LocalIndexTuple>& neighboringparticles) const
{
  // safety check
  if ((not validownedparticles_) or (not validghostedparticles_))
    FOUR_C_THROW("invalid relation of particles to bins!");

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // bin size safety check
  if (radius > binning_->minbinsize_)
    FOUR_C_THROW("the given radius is larger than the minimal bin size ({} > {})!", radius,
        binning_->minbinsize_);
#endif

  // get global id of bin
  const int gidofbin = binning_->binstrategy_->convert_pos_to_gid(position);

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // position outside computational domain
  if (gidofbin == -1) FOUR_C_THROW("position outside of computational domain!");

  // bin not owned or ghosted by this processor
  if (binning_->bincolmap_->lid(gidofbin) < 0)
    FOUR_C_THROW("position not in owned or ghosted bin on this processor!");
#endif

  // get neighboring bins to current bin
  std::vector<int> binvec;
  binning_->binstrategy_->get_neighbor_and_own_bin_ids(gidofbin, binvec);

  // iterate over neighboring bins
  for (int gidofneighborbin : binvec)
  {
    // get local id of neighboring bin
    const int collidofneighboringbin = binning_->bincolmap_->lid(gidofneighborbin);

    // neighboring bin not ghosted by this processor
    if (collidofneighboringbin < 0) continue;

    // check if current neighboring bin contains particles
    if (particlestobins_[collidofneighboringbin].empty()) continue;

    // get status of neighboring particles
    ParticleStatus neighborstatus =
        (binning_->binrowmap_->lid(gidofneighborbin) < 0) ? Ghosted : Owned;

    // iterate over particles in current neighboring bin
    for (const auto& neighborParticleIt : particlestobins_[collidofneighboringbin])
    {
      // get type of neighboring particle
      ParticleType neighbortype = neighborParticleIt.first;

      // get local index of neighboring particle
      const int neighborindex = neighborParticleIt.second;

      // get container of neighboring particle of current particle type
      ParticleContainer* neighborcontainer =
          particlecontainerbundle_->get_specific_container(neighbortype, neighborstatus);

      // get position of neighboring particle
      const double* neighborpos = neighborcontainer->get_ptr_to_state(Position, neighborindex);

      // distance vector from position to neighboring particle
      double dist[3];

      // distance between position and neighboring particle considering periodic boundaries
      distance_between_particles(position, neighborpos, dist);

      // distance between position and neighboring particle larger than radius
      if (dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2] > (radius * radius)) continue;

      // append neighboring particle
      neighboringparticles.push_back(std::make_tuple(neighbortype, neighborstatus, neighborindex));
    }
  }
}

std::array<double, 3> Particle::ParticleEngine::bin_size() const
{
  return binning_->binstrategy_->bin_size();
}

bool Particle::ParticleEngine::have_periodic_boundary_conditions() const
{
  return binning_->binstrategy_->have_periodic_boundary_conditions_applied();
}

bool Particle::ParticleEngine::have_periodic_boundary_conditions_in_spatial_direction(
    const int dim) const
{
  return binning_->binstrategy_->have_periodic_boundary_conditions_applied_in_spatial_direction(
      dim);
}

double Particle::ParticleEngine::length_of_binning_domain_in_a_spatial_direction(
    const int dim) const
{
  return binning_->binstrategy_->length_of_binning_domain_in_a_spatial_direction(dim);
}

Core::LinAlg::Matrix<3, 2> const& Particle::ParticleEngine::domain_bounding_box_corner_positions()
    const
{
  return binning_->binstrategy_->domain_bounding_box_corner_positions();
}

void Particle::ParticleEngine::distance_between_particles(
    const double* pos_i, const double* pos_j, double* r_ji) const
{
  // loop over all spatial directions
  for (int dim = 0; dim < 3; ++dim)
  {
    // vector from particle i to j
    r_ji[dim] = pos_j[dim] - pos_i[dim];

    // check for periodic boundary condition in current spatial direction
    if (binning_->binstrategy_->have_periodic_boundary_conditions_applied_in_spatial_direction(dim))
    {
      // binning domain length in current spatial direction
      double binningdomainlength =
          binning_->binstrategy_->length_of_binning_domain_in_a_spatial_direction(dim);

      // shift by periodic length if particles are closer over periodic boundary
      if (std::abs(r_ji[dim]) > (0.5 * binningdomainlength))
      {
        if (pos_i[dim] < pos_j[dim])
          r_ji[dim] -= binningdomainlength;
        else
          r_ji[dim] += binningdomainlength;
      }
    }
  }
}

std::shared_ptr<Core::IO::DiscretizationReader> Particle::ParticleEngine::bin_dis_reader(
    int restartstep) const
{
  return std::make_shared<Core::IO::DiscretizationReader>(*binning_->binstrategy_->bin_discret(),
      Global::Problem::instance()->input_control_file(), restartstep);
}

int Particle::ParticleEngine::get_number_of_particles() const
{
  int numberofparticles = 0;

  // iterate over particle types
  for (const auto& type : particlecontainerbundle_->get_particle_types())
  {
    // get container of owned particles of current particle type
    ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

    // add number of particles stored in container
    numberofparticles += container->particles_stored();
  }

  return numberofparticles;
}

int Particle::ParticleEngine::get_number_of_particles_of_specific_type(
    const ParticleType type) const
{
  if (not particlecontainerbundle_->get_particle_types().contains(type)) return 0;

  // get container of owned particles of specific particle type
  ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

  return container->particles_stored();
}

void Particle::ParticleEngine::write_bin_dis_output(const int step, const double time) const
{
  // write bins to output file
  binning_->binstrategy_->write_bin_output(step, time);
}

void Particle::ParticleEngine::init_binning_strategy()
{
  // create and init binning strategy and create bins
  Teuchos::ParameterList binning_params = Global::Problem::instance()->binning_strategy_params();
  Core::Utils::add_enum_class_to_parameter_list<Core::FE::ShapeFunctionType>(
      "spatial_approximation_type", Global::Problem::instance()->spatial_approximation_type(),
      binning_params);
  binning_ = std::make_shared<Binning>();
  binning_->binstrategy_ = std::make_shared<Core::Binstrategy::BinningStrategy>(binning_params,
      Global::Problem::instance()->output_control_file(), comm_,
      Core::Communication::my_mpi_rank(comm_));
}

void Particle::ParticleEngine::setup_binning_strategy()
{
  // determine minimum relevant bin size
  determine_min_relevant_bin_size();

  // create an initial linear distribution of row bins
  binning_->binrowmap_ = binning_->binstrategy_->create_linear_map_for_numbin(comm_);

  // initialize vector for storage of bin center coordinates and bin weights
  binning_->bincenters_ =
      std::make_shared<Core::LinAlg::MultiVector<double>>(*binning_->binrowmap_, 3);
  binning_->binweights_ =
      std::make_shared<Core::LinAlg::MultiVector<double>>(*binning_->binrowmap_, 1);

  // get all bin centers needed for repartitioning
  binning_->binstrategy_->get_all_bin_centers(*binning_->binrowmap_, *binning_->bincenters_);

  // initialize weights of all bins
  binning_->binweights_->put_scalar(1.0e-05);

  // distribute bins via recursive coordinate bisection
  binning_->binstrategy_->distribute_bins_recurs_coord_bisection(
      binning_->binrowmap_, binning_->bincenters_, binning_->binweights_);

  // create bins and fill bins into binning discretization
  binning_->binstrategy_->fill_bins_into_bin_discretization(*binning_->binrowmap_);

  // setup ghosting of bins
  setup_bin_ghosting();

  // determine bin distribution dependent maps/sets
  determine_bin_dis_dependent_maps_and_sets();

  // determine ghosting dependent maps/sets for communication
  determine_ghosting_dependent_maps_and_sets();
}

void Particle::ParticleEngine::setup_bin_ghosting()
{
  // gather bins of row map and all its neighbors (row + ghost)
  std::set<int> bins;
  for (int lid = 0; lid < binning_->binrowmap_->num_my_elements(); ++lid)
  {
    int gidofbin = binning_->binrowmap_->gid(lid);
    std::vector<int> binvec;
    // get neighboring bins
    binning_->binstrategy_->get_neighbor_and_own_bin_ids(gidofbin, binvec);
    bins.insert(binvec.begin(), binvec.end());
  }

  // remove non-existing ghost bins from original bin set
  {
    // create copy of column bins
    std::set<int> ghostbins(bins);
    // find ghost bins and check for existence
    for (int lid = 0; lid < binning_->binrowmap_->num_my_elements(); ++lid)
    {
      const int gid = binning_->binrowmap_->gid(lid);
      std::set<int>::iterator iter = ghostbins.find(gid);
      if (iter != ghostbins.end()) ghostbins.erase(iter);
    }
    // only ghost bins remain
    std::vector<int> ghostbins_vec(ghostbins.begin(), ghostbins.end());
    const int size = static_cast<int>(ghostbins.size());
    std::vector<int> pidlist(size);
    const int err = binning_->binrowmap_->remote_id_list(
        std::span(ghostbins_vec), std::span(pidlist), std::span<int>{});
    if (err < 0) FOUR_C_THROW("Core::LinAlg::Map::RemoteIDList returned err={}", err);

    for (int i = 0; i < size; ++i)
    {
      if (pidlist[i] == -1)
      {
        std::set<int>::iterator iter = bins.find(ghostbins_vec[i]);
        if (iter == bins.end()) FOUR_C_THROW("bin id is missing in bin set");
        // erase non-existing id
        bins.erase(iter);
      }
    }
  }

  // copy bin gids to a vector and create bincolmap
  std::vector<int> bincolmapvec(bins.begin(), bins.end());
  binning_->bincolmap_ = std::make_shared<Core::LinAlg::Map>(
      -1, static_cast<int>(bincolmapvec.size()), bincolmapvec.data(), 0, comm_);

  if (binning_->bincolmap_->num_global_elements() == 1 &&
      Core::Communication::num_mpi_ranks(comm_) > 1)
    FOUR_C_THROW("one bin cannot be run in parallel -> reduce BIN_SIZE_LOWER_BOUND");

  // make sure that all processors are either filled or unfilled
  binning_->binstrategy_->bin_discret()->check_filled_globally();

  // create ghosting for bins
  binning_->binstrategy_->bin_discret()->extended_ghosting(
      *binning_->bincolmap_, true, false, true, false);
}

void Particle::ParticleEngine::setup_particle_container_bundle(
    const std::map<ParticleType, std::set<ParticleState>>& particlestatestotypes) const
{
  // setup particle container bundle
  particlecontainerbundle_->setup(particlestatestotypes);
}

void Particle::ParticleEngine::setup_data_storage(
    const std::map<ParticleType, std::set<ParticleState>>& particlestatestotypes)
{
  // determine size of vectors indexed by particle types
  typevectorsize_ = ((--particlestatestotypes.end())->first) + 1;

  // allocate memory to hold particle types
  directghostingtargets_.resize(typevectorsize_);

  // allocate memory for particles being communicated to target processors
  communicatedparticletargets_.assign(
      Core::Communication::num_mpi_ranks(comm_), std::vector<int>(0));
}

void Particle::ParticleEngine::init_particle_vtp_writer()
{
  // construct and init particle runtime vtp writer
  particlevtpwriter_ = std::make_unique<ParticleRuntimeVtpWriter>(comm_);
  particlevtpwriter_->init(particlecontainerbundle_);
}

void Particle::ParticleEngine::setup_particle_vtp_writer() const
{
  // get flag to determine output of ghosted particles (debug feature)
  bool write_ghosted_particles = params_.get<bool>("WRITE_GHOSTED_PARTICLES");

  // setup particle runtime vtp writer
  particlevtpwriter_->setup(write_ghosted_particles);
}

void Particle::ParticleEngine::setup_type_weights()
{
  // allocate memory to hold particle types
  typeweights_.resize(typevectorsize_);

  // init map relating particle types to dynamic load balance factor
  std::map<ParticleType, double> typetodynloadbal;

  // read parameters relating particle types to values
  ParticleUtils::read_params_types_related_to_values(
      params_, "PHASE_TO_DYNLOADBALFAC", typetodynloadbal);

  // insert weight of particle type
  for (const auto& typeIt : typetodynloadbal) typeweights_[typeIt.first] = typeIt.second;
}

void Particle::ParticleEngine::determine_bin_dis_dependent_maps_and_sets()
{
  // clear sets and maps
  binning_->boundarybins_.clear();
  binning_->touchedbins_.clear();
  binning_->firstlayerbinsownedby_.clear();

  // check for finalized construction of binning discretization
  if (binning_->binstrategy_->bin_discret()->filled() == false)
    FOUR_C_THROW("construction of binning discretization not finalized!");

  // loop over row bins
  for (int rowlidofbin = 0; rowlidofbin < binning_->binrowmap_->num_my_elements(); ++rowlidofbin)
  {
    int currbin = binning_->binrowmap_->gid(rowlidofbin);

    // first insert all owned bins
    binning_->boundarybins_.insert(currbin);

    // get neighboring bins
    std::vector<int> binvec;
    binning_->binstrategy_->get_neighbor_bin_ids(currbin, binvec);

    // iterate over neighboring bins
    for (int neighbin : binvec)
    {
      // neighboring bin not owned by this processor
      if (binning_->binrowmap_->lid(neighbin) < 0)
      {
        // insert owned bin
        binning_->touchedbins_.insert(currbin);

        // insert owner of neighbouring bin
        int neighbinowner = binning_->binstrategy_->bin_discret()->g_element(neighbin)->owner();
        binning_->firstlayerbinsownedby_.insert(std::make_pair(neighbin, neighbinowner));
      }
    }
  }

  // determine all non-boundary bins
  std::set<int> innerbinids;

  // get number of bins in all spatial directions
  const auto binperdir = binning_->binstrategy_->bin_per_dir();

  // safety check
  for (int dim = 0; dim < 3; ++dim)
    if (binning_->binstrategy_->have_periodic_boundary_conditions_applied_in_spatial_direction(
            dim) and
        binperdir[dim] < 3)
      FOUR_C_THROW("at least 3 bins in direction with periodic boundary conditions necessary!");

  // determine range of all inner bins
  std::vector<int> ijk_min(3);
  std::vector<int> ijk_max(3);
  for (int dim = 0; dim < 3; ++dim)
  {
    ijk_min[dim] = (binperdir[dim] > 2) ? 1 : 0;
    ijk_max[dim] = (binperdir[dim] > 2) ? (binperdir[dim] - 2) : (binperdir[dim] - 1);
  }

  // ijk_range of inner bins (contains: i_min i_max j_min j_max k_min k_max)
  int ijk_range[] = {ijk_min[0], ijk_max[0], ijk_min[1], ijk_max[1], ijk_min[2], ijk_max[2]};

  // get corresponding owned bin ids in ijk range
  binning_->binstrategy_->gids_in_ijk_range(ijk_range, innerbinids, true);

  // subtract non-boundary bins from all owned bins to obtain boundary bins
  for (int currbin : innerbinids) binning_->boundarybins_.erase(currbin);
}

void Particle::ParticleEngine::determine_ghosting_dependent_maps_and_sets()
{
  // clear sets and maps
  binning_->ghostedbins_.clear();
  binning_->thisbinsghostedby_.clear();

  // check for finalized construction of binning discretization
  if (binning_->binstrategy_->bin_discret()->filled() == false)
    FOUR_C_THROW("construction of binning discretization not finalized!");

  // loop over col bins
  for (int collidofbin = 0; collidofbin < binning_->bincolmap_->num_my_elements(); ++collidofbin)
  {
    int currbin = binning_->bincolmap_->gid(collidofbin);

    // current bin not owned by this processor
    if (binning_->binrowmap_->lid(currbin) < 0) binning_->ghostedbins_.insert(currbin);
  }

  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;

  // pack data for sending
  Core::Communication::PackBuffer data;
  add_to_pack(data, binning_->ghostedbins_);

  // communicate ghosted bins between all processors
  for (int torank = 0; torank < Core::Communication::num_mpi_ranks(comm_); ++torank)
  {
    if (torank == myrank_) continue;

    sdata[torank].insert(sdata[torank].end(), data().begin(), data().end());
  }

  // communicate data via non-buffered send from proc to proc
  ParticleUtils::immediate_recv_blocking_send(comm_, sdata, rdata);

  // init receiving vector
  std::set<int> receivedbins;

  // unpack and store received data
  for (const auto& p : rdata)
  {
    const int msgsource = p.first;
    const std::vector<char>& rmsg = p.second;



    Core::Communication::UnpackBuffer buffer(rmsg);
    while (!buffer.at_end())
    {
      extract_from_pack(buffer, receivedbins);

      // iterate over received bins
      for (int receivedbin : receivedbins)
      {
        // received bin is owned by this processor
        if (binning_->binrowmap_->lid(receivedbin) >= 0)
          (binning_->thisbinsghostedby_[receivedbin]).insert(msgsource);
      }
    }
  }

  // cache ghost particle communication procs derived from bin ghosting topology (no Allreduce)
  ghost_send_procs_.clear();
  for (const auto& it : binning_->thisbinsghostedby_)
    for (int proc : it.second) ghost_send_procs_.insert(proc);

  ghost_recv_procs_.clear();
  for (int gid : binning_->ghostedbins_)
    ghost_recv_procs_.insert(binning_->binstrategy_->bin_discret()->g_element(gid)->owner());
}

void Particle::ParticleEngine::relate_half_neighboring_bins_to_owned_bins()
{
  // allocate memory for neighbors of owned bins
  binning_->halfneighboringbinstobins_.assign(
      binning_->binrowmap_->num_my_elements(), std::set<int>());

  // loop over row bins
  for (int rowlidofbin = 0; rowlidofbin < binning_->binrowmap_->num_my_elements(); ++rowlidofbin)
  {
    // get global id of bin
    const int gidofbin = binning_->binrowmap_->gid(rowlidofbin);

    // get ijk of current bin
    int ijk[3];
    binning_->binstrategy_->convert_gid_to_ijk(gidofbin, ijk);

    // get reference to neighboring bins (including current bin) of current bin
    std::set<int>& neighboringbins = binning_->halfneighboringbinstobins_[rowlidofbin];

    // insert current bin id
    neighboringbins.insert(gidofbin);

    // insert half of the surrounding bins following a specific stencil
    int ijk_range_9bin[] = {ijk[0] - 1, ijk[0] + 1, ijk[1] - 1, ijk[1] + 1, ijk[2] + 1, ijk[2] + 1};
    binning_->binstrategy_->gids_in_ijk_range(ijk_range_9bin, neighboringbins, false);

    int ijk_range_3bin[] = {ijk[0] + 1, ijk[0] + 1, ijk[1] - 1, ijk[1] + 1, ijk[2], ijk[2]};
    binning_->binstrategy_->gids_in_ijk_range(ijk_range_3bin, neighboringbins, false);

    int ijk_range_1bin[] = {ijk[0], ijk[0], ijk[1] + 1, ijk[1] + 1, ijk[2], ijk[2]};
    binning_->binstrategy_->gids_in_ijk_range(ijk_range_1bin, neighboringbins, false);
  }

  // iterate over bins being ghosted on this processor
  for (int gidofbin : binning_->ghostedbins_)
  {
    // get neighboring bins
    std::vector<int> binvec;
    binning_->binstrategy_->get_neighbor_bin_ids(gidofbin, binvec);

    // iterate over neighboring bins
    for (int neighbin : binvec)
    {
      // get local id of bin
      const int rowlidofbin = binning_->binrowmap_->lid(neighbin);

      // neighboring bin not owned by this processor
      if (rowlidofbin < 0) continue;

      // insert neighboring bins being ghosted on this processor
      binning_->halfneighboringbinstobins_[rowlidofbin].insert(gidofbin);
    }
  }

  // validate flag denoting valid relation of half surrounding neighboring bins to owned bins
  binning_->validhalfneighboringbins_ = true;
}

void Particle::ParticleEngine::check_particles_at_boundaries(
    std::vector<std::set<int>>& particlestoremove) const
{
  // safety check
  if (not validownedparticles_) FOUR_C_THROW("invalid relation of owned particles to bins!");

  // get bounding box dimensions
  Core::LinAlg::Matrix<3, 2> boundingbox =
      binning_->binstrategy_->domain_bounding_box_corner_positions();

  // count particles that left the computational domain
  int numparticlesoutside = 0;

  // iterate over owned bins at the boundary
  for (int bdrybin : binning_->boundarybins_)
  {
    // get local id of bin
    const int collidofbin = binning_->bincolmap_->lid(bdrybin);

    // check if current bin contains owned particles
    if (particlestobins_[collidofbin].empty()) continue;

    // iterate over owned particles in current bin
    for (const auto& particleIt : particlestobins_[collidofbin])
    {
      // get type of particle
      ParticleType type = particleIt.first;

      // get local index of owned particle
      const int ownedindex = particleIt.second;

      // get container of owned particle of current particle type
      ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

      // get position of particle
      double* currpos = container->get_ptr_to_state(Position, ownedindex);

      // get global id of bin
      const int gidofbin = binning_->binstrategy_->convert_pos_to_gid(currpos);

      // particle left computational domain
      if (gidofbin == -1)
      {
        (particlestoremove[type]).insert(ownedindex);

        // get global id of particle
        const int* currglobalid = container->get_ptr_to_global_id(ownedindex);

#ifdef FOUR_C_ENABLE_ASSERTIONS
        if (currglobalid[0] < 0) FOUR_C_THROW("no global id assigned to particle!");
#endif

        // insert freed global id
        particleuniqueglobalidhandler_->insert_freed_global_id(currglobalid[0]);

        ++numparticlesoutside;

        continue;
      }

      // no periodic boundary conditions
      if (not binning_->binstrategy_->have_periodic_boundary_conditions_applied()) continue;

      // check for periodic boundary in each spatial directions
      for (int dim = 0; dim < 3; ++dim)
      {
        if (binning_->binstrategy_->have_periodic_boundary_conditions_applied_in_spatial_direction(
                dim))
        {
          // binning domain length in current spatial direction
          double binningdomainlength =
              binning_->binstrategy_->length_of_binning_domain_in_a_spatial_direction(dim);

          // shift position by periodic length
          if (currpos[dim] < boundingbox(dim, 0))
            currpos[dim] += binningdomainlength;
          else if (currpos[dim] > boundingbox(dim, 1))
            currpos[dim] -= binningdomainlength;
        }
      }
    }
  }

  // short screen output
  if (numparticlesoutside)
    Core::IO::cout << "on processor " << myrank_ << " removed " << numparticlesoutside
                   << " particle(s) being outside the computational domain!" << Core::IO::endl;
}

void Particle::ParticleEngine::determine_particles_to_be_distributed(
    std::vector<ParticleObjShrdPtr>& particlestodistribute,
    std::vector<std::vector<ParticleObjShrdPtr>>& particlestosend,
    std::vector<std::vector<std::pair<int, ParticleObjShrdPtr>>>& particlestokeep)
{
  // clear particles being communicated to target processors
  communicatedparticletargets_.assign(
      Core::Communication::num_mpi_ranks(comm_), std::vector<int>(0));

  // number of particles to distribute
  int numparticles = particlestodistribute.size();

  // global ids of bins containing particles
  std::vector<int> bingidlist(numparticles);

  // iterate over particles objects
  for (int i = 0; i < numparticles; ++i)
  {
    // get states of particle
    const ParticleStates& states = particlestodistribute[i]->return_particle_states();

    // get position of particle
    const std::vector<double>& pos = states[Position];

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // get type of particles
    ParticleType type = particlestodistribute[i]->return_particle_type();

    // get container of owned particles of current particle type
    ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

    if (static_cast<int>(pos.size()) != container->get_state_dim(Position))
      FOUR_C_THROW("dimension of particle state '{}' not valid!", enum_to_state_name(Position));
#endif

    // get global id of bin
    bingidlist[i] = binning_->binstrategy_->convert_pos_to_gid(pos.data());
  }

  // get corresponding processor id
  std::vector<int> pidlist(numparticles);
  {
    // only unique id lists are accepted in RemoteIDList
    // 1) make gid list unique
    std::set<int> unique_bingidlist(bingidlist.begin(), bingidlist.end());
    std::vector<int> uniquevec_bingidlist(unique_bingidlist.begin(), unique_bingidlist.end());
    const int uniquesize = static_cast<int>(uniquevec_bingidlist.size());

    // 2) communication
    std::vector<int> unique_pidlist(uniquesize);
    int err = binning_->binrowmap_->remote_id_list(
        std::span<int>(uniquevec_bingidlist), std::span<int>(unique_pidlist), std::span<int>{});
    if (err < 0) FOUR_C_THROW("RemoteIDList returned err={}", err);

    // 3) build full pid list via lookup table
    std::map<int, int> lookuptable;
    for (int s = 0; s < uniquesize; ++s)
      lookuptable.insert(
          lookuptable.end(), std::pair<int, int>(uniquevec_bingidlist[s], unique_pidlist[s]));
    for (int s = 0; s < numparticles; ++s) pidlist[s] = lookuptable[bingidlist[s]];
  }

  // count particles that are outside of the computational domain
  int numparticlesoutside = 0;

  // iterate over particles objects
  for (int i = 0; i < numparticles; ++i)
  {
    // get type of particle
    ParticleType type = particlestodistribute[i]->return_particle_type();

    // get owner of particle
    int ownerofparticle = pidlist[i];

    // particle outside of computational domain
    if (ownerofparticle == -1)
    {
      ++numparticlesoutside;

#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (particlestodistribute[i]->return_particle_global_id() < 0)
        FOUR_C_THROW("no global id assigned to particle!");
#endif

      // insert freed global id
      particleuniqueglobalidhandler_->insert_freed_global_id(
          particlestodistribute[i]->return_particle_global_id());
    }
    // particle is owned by this processor
    else if (myrank_ == ownerofparticle)
      particlestokeep[type].push_back(std::make_pair(ownerofparticle, particlestodistribute[i]));
    // particle is owned by another processor
    else
    {
      // append particle to be send
      particlestosend[ownerofparticle].push_back(particlestodistribute[i]);

      // append global id of particle to be send
      communicatedparticletargets_[ownerofparticle].emplace_back(
          particlestodistribute[i]->return_particle_global_id());
    }
  }

  // short screen output
  if (numparticlesoutside)
    Core::IO::cout << "on processor " << myrank_ << " removed " << numparticlesoutside
                   << " particle(s) being outside the computational domain!" << Core::IO::endl;

  // clear after all particles are prepared for distribution
  particlestodistribute.clear();
}

void Particle::ParticleEngine::determine_particles_to_be_transferred(
    std::vector<std::set<int>>& particlestoremove,
    std::vector<std::vector<ParticleObjShrdPtr>>& particlestosend)
{
  // safety check
  if (not validownedparticles_) FOUR_C_THROW("invalid relation of owned particles to bins!");

  // clear particles being communicated to target processors
  communicatedparticletargets_.assign(
      Core::Communication::num_mpi_ranks(comm_), std::vector<int>(0));

  // iterate over this processors bins being touched by other processors
  for (int touchedbin : binning_->touchedbins_)
  {
    // get local id of bin
    const int collidofbin = binning_->bincolmap_->lid(touchedbin);

    // check if current bin contains owned particles
    if (particlestobins_[collidofbin].empty()) continue;

    // iterate over owned particles in current bin
    for (const auto& particleIt : particlestobins_[collidofbin])
    {
      // get type of particle
      ParticleType type = particleIt.first;

      // get local index of owned particle
      const int ownedindex = particleIt.second;

      // get container of owned particle of current particle type
      ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

      // get position of particle
      const double* currpos = container->get_ptr_to_state(Position, ownedindex);

      // get global id of bin
      const int gidofbin = binning_->binstrategy_->convert_pos_to_gid(currpos);

      // particle left computational domain
      if (gidofbin == -1)
      {
#ifdef FOUR_C_ENABLE_ASSERTIONS
        if (not particlestoremove[type].contains(ownedindex))
          FOUR_C_THROW(
              "on processor {} a particle left the computational domain without being detected!",
              myrank_);
#endif
        continue;
      }

      // particle remains owned on this processor
      if (binning_->binrowmap_->lid(gidofbin) >= 0) continue;

      // get owning processor
      auto targetIt = binning_->firstlayerbinsownedby_.find(gidofbin);
      if (targetIt == binning_->firstlayerbinsownedby_.end())
        FOUR_C_THROW("particle not owned on this proc but target processor is unknown!");
      int sendtoproc = targetIt->second;

      int globalid(0);
      ParticleStates states;
      container->get_particle(ownedindex, globalid, states);

      // append particle to be send
      particlestosend[sendtoproc].emplace_back(
          std::make_shared<ParticleObject>(type, globalid, states, gidofbin));

      // store index of particle to be removed from containers after particle transfer
      (particlestoremove[type]).insert(ownedindex);

      // append global id of particle to be send
      communicatedparticletargets_[sendtoproc].emplace_back(globalid);
    }
  }
}

void Particle::ParticleEngine::pack_particles_to_be_ghosted(
    std::map<int, std::vector<char>>& sdata) const
{
  // safety check
  if (not validownedparticles_) FOUR_C_THROW("invalid relation of owned particles to bins!");

  // iterate over this processors bins being ghosted by other processors
  for (const auto& targetIt : binning_->thisbinsghostedby_)
  {
    // bin being ghosted on other processors
    const int ghostedbin = targetIt.first;

    // get local id of bin
    const int collidofbin = binning_->bincolmap_->lid(ghostedbin);

    // check if current bin contains owned particles
    if (particlestobins_[collidofbin].empty()) continue;

    // iterate over owned particles in current bin
    for (const auto& particleIt : particlestobins_[collidofbin])
    {
      // get type of particle
      ParticleType type = particleIt.first;

      // get local index of owned particle
      const int ownedindex = particleIt.second;

      // get container of owned particle of current particle type
      ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

      int globalid(0);
      ParticleStates states;
      container->get_particle(ownedindex, globalid, states);

      // pre-pack header (type, globalid, bingid, ownedindex) once per particle
      Core::Communication::PackBuffer header;
      add_to_pack(header, static_cast<int>(type));
      add_to_pack(header, globalid);
      add_to_pack(header, ghostedbin);
      add_to_pack(header, ownedindex);

      // pre-pack states once, reused for all target processors
      Core::Communication::PackBuffer statesdata;
      add_to_pack(statesdata, states);

      // iterate over target processors and append packed data
      for (int sendtoproc : targetIt.second)
      {
        auto& buf = sdata[sendtoproc];
        buf.insert(buf.end(), header().begin(), header().end());
        buf.insert(buf.end(), statesdata().begin(), statesdata().end());
      }
    }
  }
}

void Particle::ParticleEngine::pack_particles_to_be_refreshed(
    std::map<int, std::vector<char>>& sdata) const
{
  // safety check
  if (not validdirectghosting_) FOUR_C_THROW("invalid direct ghosting!");

  // iterate over particle types
  for (const auto& type : particlecontainerbundle_->get_particle_types())
  {
    // check for particles of current type to be sent
    if (directghostingtargets_[type].empty()) continue;

    // get container of owned particles of current particle type
    ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

    // iterate over owned particles of current type
    for (const auto& indexIt : directghostingtargets_[type])
    {
      int ownedindex = indexIt.first;

      // get particle states once
      int globalid(0);
      ParticleStates states;
      container->get_particle(ownedindex, globalid, states);

      // pre-pack states data (reused for all targets of this particle)
      Core::Communication::PackBuffer statesdata;
      add_to_pack(statesdata, states);

      // iterate over target processors
      for (const auto& targetIt : indexIt.second)
      {
        int sendtoproc = targetIt.first;
        int ghostedindex = targetIt.second;

        // pack type and ghosted index header
        Core::Communication::PackBuffer header;
        add_to_pack(header, static_cast<int>(type));
        add_to_pack(header, ghostedindex);

        // append header + pre-packed states to send buffer
        auto& buf = sdata[sendtoproc];
        buf.insert(buf.end(), header().begin(), header().end());
        buf.insert(buf.end(), statesdata().begin(), statesdata().end());
      }
    }
  }
}

void Particle::ParticleEngine::pack_specific_states_of_particles_to_be_refreshed(
    const StatesOfTypesToRefresh& particlestatestotypes,
    std::map<int, std::vector<char>>& sdata) const
{
  // safety check
  if (not validdirectghosting_) FOUR_C_THROW("invalid direct ghosting!");

  // iterate over particle types
  for (const auto& typeIt : particlestatestotypes)
  {
    // get type of particles
    ParticleType type = typeIt.first;

    // check for particles of current type to be sent
    if (directghostingtargets_[type].empty()) continue;

    // determine necessary size of vector for states
    int statesvectorsize = *std::max_element(typeIt.second.begin(), typeIt.second.end()) + 1;

    // get container of owned particles of current particle type
    ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

    // iterate over owned particles of current type
    for (const auto& indexIt : directghostingtargets_[type])
    {
      int ownedindex = indexIt.first;

      // allocate memory to hold particle states
      ParticleStates states;
      states.assign(statesvectorsize, std::vector<double>(0));

      // iterate over states to be sent
      for (const auto& state : typeIt.second)
      {
        // get particle state dimension
        int statedim = container->get_state_dim(state);

        // get pointer to particle state
        const double* state_ptr = container->get_ptr_to_state(state, ownedindex);

        // fill particle state
        states[state].assign(state_ptr, state_ptr + statedim);
      }

      // pre-pack states data (reused for all targets of this particle)
      Core::Communication::PackBuffer statesdata;
      add_to_pack(statesdata, states);

      // iterate over target processors
      for (const auto& targetIt : indexIt.second)
      {
        int sendtoproc = targetIt.first;
        int ghostedindex = targetIt.second;

        // pack type and ghosted index header
        Core::Communication::PackBuffer header;
        add_to_pack(header, static_cast<int>(type));
        add_to_pack(header, ghostedindex);

        // append header + pre-packed states to send buffer
        auto& buf = sdata[sendtoproc];
        buf.insert(buf.end(), header().begin(), header().end());
        buf.insert(buf.end(), statesdata().begin(), statesdata().end());
      }
    }
  }
}

void Particle::ParticleEngine::communicate_particles(
    std::vector<std::vector<ParticleObjShrdPtr>>& particlestosend,
    std::vector<std::vector<std::pair<int, ParticleObjShrdPtr>>>& particlestoreceive) const
{
  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;

  // pack data for sending
  for (int torank = 0; torank < Core::Communication::num_mpi_ranks(comm_); ++torank)
  {
    if (particlestosend[torank].empty()) continue;

    for (const auto& iter : particlestosend[torank])
    {
      Core::Communication::PackBuffer data;
      iter->pack(data);
      sdata[torank].insert(sdata[torank].end(), data().begin(), data().end());
    }
  }

  // clear after all particles are packed
  particlestosend.clear();

  // communicate data via non-buffered send from proc to proc
  ParticleUtils::immediate_recv_blocking_send(comm_, sdata, rdata);

  // unpack and store received data
  for (const auto& p : rdata)
  {
    const int msgsource = p.first;
    const std::vector<char>& rmsg = p.second;



    Core::Communication::UnpackBuffer buffer(rmsg);
    while (!buffer.at_end())
    {
      std::shared_ptr<Core::Communication::ParObject> object(Core::Communication::factory(buffer));
      ParticleObjShrdPtr particleobject = std::dynamic_pointer_cast<ParticleObject>(object);
      if (particleobject == nullptr) FOUR_C_THROW("received object is not a particle object!");

      // store received particle
      particlestoreceive[particleobject->return_particle_type()].push_back(
          std::make_pair(msgsource, particleobject));
    }
  }
}

void Particle::ParticleEngine::communicate_refreshed_particles(
    std::map<int, std::vector<char>>& sdata, const std::vector<int>* known_recv_sizes,
    std::vector<int>* out_recv_sizes) const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::ParticleEngine::communicate_refreshed_particles");

  const int numsendtoprocs = static_cast<int>(refresh_send_procs_.size());
  const int numrecvfromprocs = static_cast<int>(refresh_recv_procs_.size());

  std::vector<int> recvsources(refresh_recv_procs_.begin(), refresh_recv_procs_.end());
  std::vector<int> msgsizestorecv(numrecvfromprocs);

  // Phase 1: size exchange — skipped when recv sizes are already cached
  std::vector<MPI_Request> sizesendrequest;
  if (known_recv_sizes != nullptr)
  {
    // use pre-cached recv sizes; no round-trip needed
    msgsizestorecv = *known_recv_sizes;
  }
  else
  {
    // send size of messages to ALL known refresh targets (0 if no data)
    sizesendrequest.resize(numsendtoprocs);
    std::vector<int> msgsizestosend(numsendtoprocs);
    {
      int counter = 0;
      for (int torank : refresh_send_procs_)
      {
        auto it = sdata.find(torank);
        msgsizestosend[counter] = (it != sdata.end()) ? static_cast<int>(it->second.size()) : 0;
        MPI_Isend(
            &msgsizestosend[counter], 1, MPI_INT, torank, 1234, comm_, &sizesendrequest[counter]);
        ++counter;
      }
    }

    // receive size of messages from ALL known senders
    std::vector<MPI_Request> sizerecvrequest(numrecvfromprocs);
    for (int i = 0; i < numrecvfromprocs; ++i)
      MPI_Irecv(&msgsizestorecv[i], 1, MPI_INT, recvsources[i], 1234, comm_, &sizerecvrequest[i]);

    MPI_Waitall(numrecvfromprocs, sizerecvrequest.data(), MPI_STATUSES_IGNORE);

    // hand sizes back to caller for caching
    if (out_recv_sizes != nullptr) *out_recv_sizes = msgsizestorecv;
  }

  // Phase 2: post data receives using now-known sizes
  std::map<int, std::vector<char>> rdata;
  std::vector<MPI_Request> recvrequest(numrecvfromprocs);
  for (int i = 0; i < numrecvfromprocs; ++i)
  {
    if (msgsizestorecv[i] > 0)
    {
      rdata[recvsources[i]].resize(msgsizestorecv[i]);
      MPI_Irecv(rdata[recvsources[i]].data(), msgsizestorecv[i], MPI_CHAR, recvsources[i], 5678,
          comm_, &recvrequest[i]);
    }
    else
    {
      recvrequest[i] = MPI_REQUEST_NULL;
    }
  }

  // wait for size sends to complete (if Phase 1 was active), then send data
  if (!sizesendrequest.empty())
    MPI_Waitall(
        static_cast<int>(sizesendrequest.size()), sizesendrequest.data(), MPI_STATUSES_IGNORE);

  std::vector<MPI_Request> sendrequest;
  sendrequest.reserve(sdata.size());
  for (auto& p : sdata)
  {
    sendrequest.emplace_back();
    MPI_Isend(p.second.data(), static_cast<int>(p.second.size()), MPI_CHAR, p.first, 5678, comm_,
        &sendrequest.back());
  }

  // wait for completion
  {
    TEUCHOS_FUNC_TIME_MONITOR(
        "Particle::ParticleEngine::communicate_refreshed_particles::wait_for_completion");
    if (!sendrequest.empty())
      MPI_Waitall(static_cast<int>(sendrequest.size()), sendrequest.data(), MPI_STATUSES_IGNORE);
    sdata.clear();
    MPI_Waitall(numrecvfromprocs, recvrequest.data(), MPI_STATUSES_IGNORE);
  }

  // unpack received data directly into ghosted particle containers
  for (const auto& p : rdata)
  {
    const std::vector<char>& rmsg = p.second;

    Core::Communication::UnpackBuffer buffer(rmsg);
    while (!buffer.at_end())
    {
      // unpack particle type
      int type_int;
      extract_from_pack(buffer, type_int);
      ParticleType type = static_cast<ParticleType>(type_int);

      // unpack ghosted index
      int ghostedindex;
      extract_from_pack(buffer, ghostedindex);

      // unpack states
      ParticleStates states;
      extract_from_pack(buffer, states);

      // replace particle directly in container of ghosted particles
      ParticleContainer* container =
          particlecontainerbundle_->get_specific_container(type, Ghosted);
      container->replace_particle(ghostedindex, -1, states);
    }
  }
}

std::vector<int> Particle::ParticleEngine::compute_refresh_specific_recv_sizes(
    const StatesOfTypesToRefresh& particlestatestotypes) const
{
  // Determine the deterministic per-particle pack size for each particle type that appears in
  // the requested pattern. This must mirror exactly what pack_specific_states_of_particles_to_-
  // be_refreshed writes per particle: a header [int type, int ghostedindex] followed by a
  // ParticleStates vector with the requested states filled (others empty).
  std::map<ParticleType, int> per_particle_size;
  for (const auto& [type, stateset] : particlestatestotypes)
  {
    if (stateset.empty())
    {
      per_particle_size[type] = 0;
      continue;
    }

    const int statesvectorsize = *std::max_element(stateset.begin(), stateset.end()) + 1;

    ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

    // build a single dummy particle states vector matching the pack layout used by
    // pack_specific_states_of_particles_to_be_refreshed
    ParticleStates dummy;
    dummy.assign(statesvectorsize, std::vector<double>(0));
    for (const auto& state : stateset) dummy[state].assign(container->get_state_dim(state), 0.0);

    Core::Communication::PackBuffer statesdata;
    add_to_pack(statesdata, dummy);

    Core::Communication::PackBuffer header;
    add_to_pack(header, static_cast<int>(type));
    add_to_pack(header, 0);  // dummy ghosted index

    per_particle_size[type] = static_cast<int>(header().size() + statesdata().size());
  }

  // sum up bytes received from each sender as Sum_type ghost_count[sender][type] * size_per[type]
  std::vector<int> recv_sizes;
  recv_sizes.reserve(refresh_recv_procs_.size());
  for (int sender : refresh_recv_procs_)
  {
    int bytes = 0;
    auto pit = ghost_count_from_proc_.find(sender);
    if (pit != ghost_count_from_proc_.end())
    {
      for (const auto& [type, size_per] : per_particle_size)
      {
        auto cit = pit->second.find(type);
        if (cit != pit->second.end()) bytes += cit->second * size_per;
      }
    }
    recv_sizes.push_back(bytes);
  }
  return recv_sizes;
}

std::vector<int> Particle::ParticleEngine::compute_refresh_all_states_recv_sizes() const
{
  // Build the per-type deterministic per-particle pack size that
  // pack_particles_to_be_refreshed produces: a header [int type, int ghostedindex]
  // followed by the full ParticleStates of the type. We reconstruct the exact
  // ParticleStates layout that ParticleContainer::get_particle would yield using
  // the container's set of stored states and their dimensions.
  std::map<ParticleType, int> per_particle_size;
  for (const auto& type : particlecontainerbundle_->get_particle_types())
  {
    ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);
    const std::set<ParticleState>& storedstates = container->get_stored_states();

    if (storedstates.empty())
    {
      per_particle_size[type] = 0;
      continue;
    }

    const int statesvectorsize = *std::max_element(storedstates.begin(), storedstates.end()) + 1;

    ParticleStates dummy;
    dummy.assign(statesvectorsize, std::vector<double>(0));
    for (const auto& state : storedstates)
      dummy[state].assign(container->get_state_dim(state), 0.0);

    Core::Communication::PackBuffer statesdata;
    add_to_pack(statesdata, dummy);

    Core::Communication::PackBuffer header;
    add_to_pack(header, static_cast<int>(type));
    add_to_pack(header, 0);  // dummy ghosted index

    per_particle_size[type] = static_cast<int>(header().size() + statesdata().size());
  }

  // sum bytes received from each sender as Sum_type ghost_count[sender][type] * size_per[type]
  std::vector<int> recv_sizes;
  recv_sizes.reserve(refresh_recv_procs_.size());
  for (int sender : refresh_recv_procs_)
  {
    int bytes = 0;
    auto pit = ghost_count_from_proc_.find(sender);
    if (pit != ghost_count_from_proc_.end())
    {
      for (const auto& [type, size_per] : per_particle_size)
      {
        auto cit = pit->second.find(type);
        if (cit != pit->second.end()) bytes += cit->second * size_per;
      }
    }
    recv_sizes.push_back(bytes);
  }
  return recv_sizes;
}

void Particle::ParticleEngine::communicate_direct_ghosting_map(
    std::map<int, std::map<ParticleType, std::map<int, std::pair<int, int>>>>& directghosting)
{
  // iterate over particle types
  for (const auto& type : particlecontainerbundle_->get_particle_types())
    directghostingtargets_[type].clear();

  // invalidate flags denoting validity of direct ghosting
  validdirectghosting_ = false;

  // cache procs from which we receive refreshed particle data: these are the procs that sent
  // ghost particles to us (keys of directghosting map)
  refresh_recv_procs_.clear();
  for (const auto& p : directghosting) refresh_recv_procs_.insert(p.first);

  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;

  // pack data for sending
  for (const auto& p : directghosting)
  {
    Core::Communication::PackBuffer data;
    add_to_pack(data, p.second);
    std::swap(sdata[p.first], data());
  }

  // clear after all ghosting information is packed
  directghosting.clear();

  // communicate direct ghosting map using cached ghost comm graph (no Allreduce)
  // send to ghost_recv_procs_ (procs that sent us ghost data); receive from ghost_send_procs_
  const int numsendtoprocs_dgm = static_cast<int>(ghost_recv_procs_.size());
  const int numrecvfromprocs_dgm = static_cast<int>(ghost_send_procs_.size());

  std::vector<MPI_Request> sizesendrequest_dgm(numsendtoprocs_dgm);
  std::vector<int> msgsizestosend_dgm(numsendtoprocs_dgm);
  {
    int counter = 0;
    for (int torank : ghost_recv_procs_)
    {
      auto it = sdata.find(torank);
      msgsizestosend_dgm[counter] = (it != sdata.end()) ? static_cast<int>(it->second.size()) : 0;
      MPI_Isend(&msgsizestosend_dgm[counter], 1, MPI_INT, torank, 1234, comm_,
          &sizesendrequest_dgm[counter]);
      ++counter;
    }
  }

  std::vector<int> recvsources_dgm(ghost_send_procs_.begin(), ghost_send_procs_.end());
  std::vector<int> msgsizestorecv_dgm(numrecvfromprocs_dgm);
  std::vector<MPI_Request> sizerecvrequest_dgm(numrecvfromprocs_dgm);
  for (int i = 0; i < numrecvfromprocs_dgm; ++i)
    MPI_Irecv(&msgsizestorecv_dgm[i], 1, MPI_INT, recvsources_dgm[i], 1234, comm_,
        &sizerecvrequest_dgm[i]);

  MPI_Waitall(numrecvfromprocs_dgm, sizerecvrequest_dgm.data(), MPI_STATUSES_IGNORE);

  std::vector<MPI_Request> recvrequest_dgm(numrecvfromprocs_dgm);
  for (int i = 0; i < numrecvfromprocs_dgm; ++i)
  {
    if (msgsizestorecv_dgm[i] > 0)
    {
      rdata[recvsources_dgm[i]].resize(msgsizestorecv_dgm[i]);
      MPI_Irecv(rdata[recvsources_dgm[i]].data(), msgsizestorecv_dgm[i], MPI_CHAR,
          recvsources_dgm[i], 5678, comm_, &recvrequest_dgm[i]);
    }
    else
    {
      recvrequest_dgm[i] = MPI_REQUEST_NULL;
    }
  }

  MPI_Waitall(numsendtoprocs_dgm, sizesendrequest_dgm.data(), MPI_STATUSES_IGNORE);

  std::vector<MPI_Request> sendrequest_dgm;
  sendrequest_dgm.reserve(sdata.size());
  for (auto& p : sdata)
  {
    sendrequest_dgm.emplace_back();
    MPI_Isend(p.second.data(), static_cast<int>(p.second.size()), MPI_CHAR, p.first, 5678, comm_,
        &sendrequest_dgm.back());
  }

  if (!sendrequest_dgm.empty())
    MPI_Waitall(
        static_cast<int>(sendrequest_dgm.size()), sendrequest_dgm.data(), MPI_STATUSES_IGNORE);
  sdata.clear();
  MPI_Waitall(numrecvfromprocs_dgm, recvrequest_dgm.data(), MPI_STATUSES_IGNORE);

  // init receiving map
  std::map<ParticleType, std::map<int, std::pair<int, int>>> receiveddirectghosting;

  // unpack and store received data
  for (const auto& p : rdata)
  {
    const std::vector<char>& rmsg = p.second;

    Core::Communication::UnpackBuffer buffer(rmsg);
    while (!buffer.at_end())
    {
      extract_from_pack(buffer, receiveddirectghosting);

      // iterate over particle types
      for (const auto& typeIt : receiveddirectghosting)
      {
        // get type of particles
        ParticleType type = typeIt.first;

        // iterate over this processors local indices of owned particles
        for (const auto& indexIt : typeIt.second)
        {
          // get index of owned particle
          int ownedindex = indexIt.first;

          (directghostingtargets_[type])[ownedindex].push_back(indexIt.second);
        }
      }
    }
  }

  // validate flags denoting validity of direct ghosting
  validdirectghosting_ = true;

  // invalidate cached recv sizes — ghosting topology (and thus message sizes) has changed
  refresh_specific_recv_sizes_cache_.clear();
  refresh_all_states_recv_sizes_cache_.clear();

  // cache procs to which we send refreshed particle data
  refresh_send_procs_.clear();
  for (const auto& type : particlecontainerbundle_->get_particle_types())
    for (const auto& [ownedindex, targets] : directghostingtargets_[type])
      for (const auto& [proc, ghostedindex] : targets) refresh_send_procs_.insert(proc);
}

void Particle::ParticleEngine::insert_owned_particles(
    std::vector<std::vector<std::pair<int, ParticleObjShrdPtr>>>& particlestoinsert)
{
  // iterate over particle types
  for (const auto& type : particlecontainerbundle_->get_particle_types())
  {
    // check for particles of current type
    if (particlestoinsert[type].empty()) continue;

    // get container of owned particles of current particle type
    ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

    // iterate over particle objects pairs
    for (const auto& objectpair : particlestoinsert[type])
    {
      // get particle object
      ParticleObjShrdPtr particleobject = objectpair.second;

      // get global id of particle
      int globalid = particleobject->return_particle_global_id();

      // get states of particle
      const ParticleStates& states = particleobject->return_particle_states();

#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (globalid < 0) FOUR_C_THROW("no global id assigned to particle!");

      // get bin of particle
      int gidofbin = particleobject->return_bin_gid();

      // bin particle
      if (gidofbin < 0)
      {
        // get position of particle
        const std::vector<double>& pos = states[Position];

        // get type of particles
        ParticleType type = particleobject->return_particle_type();

        // get container of owned particles of current particle type
        ParticleContainer* container =
            particlecontainerbundle_->get_specific_container(type, Owned);

        if (static_cast<int>(pos.size()) != container->get_state_dim(Position))
          FOUR_C_THROW("dimension of particle state '{}' not valid!", enum_to_state_name(Position));

        // get global id of bin
        gidofbin = binning_->binstrategy_->convert_pos_to_gid(pos.data());
      }

      // particle not owned by this processor
      if (binning_->binrowmap_->lid(gidofbin) < 0)
        FOUR_C_THROW("particle received not owned on this proc!");
#endif

      // add particle to container of owned particles
      int index(0);
      container->add_particle(index, globalid, states);
    }
  }

  // clear after all particles are inserted
  particlestoinsert.clear();

  // invalidate particle safety flags
  invalidate_particle_safety_flags();
}

void Particle::ParticleEngine::communicate_and_insert_ghost_particles(
    std::map<int, std::vector<char>>& sdata,
    std::map<int, std::map<ParticleType, std::map<int, std::pair<int, int>>>>& directghosting)
{
  // reset per-(sender,type) ghost counts; they will be re-populated during unpack below
  ghost_count_from_proc_.clear();

  const int numsendtoprocs = static_cast<int>(ghost_send_procs_.size());
  const int numrecvfromprocs = static_cast<int>(ghost_recv_procs_.size());

  // send size of messages to ALL known ghost targets (0 if no data)
  std::vector<MPI_Request> sizesendrequest(numsendtoprocs);
  std::vector<int> msgsizestosend(numsendtoprocs);
  {
    int counter = 0;
    for (int torank : ghost_send_procs_)
    {
      auto it = sdata.find(torank);
      msgsizestosend[counter] = (it != sdata.end()) ? static_cast<int>(it->second.size()) : 0;
      MPI_Isend(
          &msgsizestosend[counter], 1, MPI_INT, torank, 1234, comm_, &sizesendrequest[counter]);
      ++counter;
    }
  }

  // receive size of messages from ALL known senders
  std::vector<int> recvsources(ghost_recv_procs_.begin(), ghost_recv_procs_.end());
  std::vector<int> msgsizestorecv(numrecvfromprocs);
  std::vector<MPI_Request> sizerecvrequest(numrecvfromprocs);
  for (int i = 0; i < numrecvfromprocs; ++i)
  {
    MPI_Irecv(&msgsizestorecv[i], 1, MPI_INT, recvsources[i], 1234, comm_, &sizerecvrequest[i]);
  }

  // wait for all size receives to complete
  MPI_Waitall(numrecvfromprocs, sizerecvrequest.data(), MPI_STATUSES_IGNORE);

  // post receives for actual data from senders with non-zero size
  std::map<int, std::vector<char>> rdata;
  std::vector<MPI_Request> recvrequest(numrecvfromprocs);
  for (int i = 0; i < numrecvfromprocs; ++i)
  {
    if (msgsizestorecv[i] > 0)
    {
      rdata[recvsources[i]].resize(msgsizestorecv[i]);
      MPI_Irecv(rdata[recvsources[i]].data(), msgsizestorecv[i], MPI_CHAR, recvsources[i], 5678,
          comm_, &recvrequest[i]);
    }
    else
    {
      recvrequest[i] = MPI_REQUEST_NULL;
    }
  }

  // wait for size sends to complete, then send data
  MPI_Waitall(numsendtoprocs, sizesendrequest.data(), MPI_STATUSES_IGNORE);

  std::vector<MPI_Request> sendrequest;
  sendrequest.reserve(sdata.size());
  for (auto& p : sdata)
  {
    sendrequest.emplace_back();
    MPI_Isend(p.second.data(), static_cast<int>(p.second.size()), MPI_CHAR, p.first, 5678, comm_,
        &sendrequest.back());
  }

  // wait for completion
  if (!sendrequest.empty())
    MPI_Waitall(static_cast<int>(sendrequest.size()), sendrequest.data(), MPI_STATUSES_IGNORE);
  sdata.clear();
  MPI_Waitall(numrecvfromprocs, recvrequest.data(), MPI_STATUSES_IGNORE);

  // unpack received data directly into ghosted particle containers
  for (const auto& p : rdata)
  {
    const int sendingproc = p.first;
    Core::Communication::UnpackBuffer buffer(p.second);
    while (!buffer.at_end())
    {
      // unpack particle header
      int type_int, globalid, bingid, ownedindex;
      extract_from_pack(buffer, type_int);
      extract_from_pack(buffer, globalid);
      extract_from_pack(buffer, bingid);
      extract_from_pack(buffer, ownedindex);

      // unpack states
      ParticleStates states;
      extract_from_pack(buffer, states);

      ParticleType type = static_cast<ParticleType>(type_int);

      // get container of ghosted particles of current particle type
      ParticleContainer* container =
          particlecontainerbundle_->get_specific_container(type, Ghosted);

      // add particle to container of ghosted particles
      int ghostedindex(0);
      container->add_particle(ghostedindex, globalid, states);

      // add index relating (owned and ghosted) particles to col bins
      particlestobins_[binning_->bincolmap_->lid(bingid)].push_back(
          std::make_pair(type, ghostedindex));

      // record ghost particle count per (sender proc, particle type) for later use in
      // computing refresh-specific recv sizes locally without an MPI round-trip
      ++ghost_count_from_proc_[sendingproc][type];

      // insert necessary information for direct ghosting map
      (((directghosting[sendingproc])[type])[ownedindex]) = std::make_pair(myrank_, ghostedindex);
    }
  }

  // validate flag denoting valid relation of ghosted particles to bins
  validghostedparticles_ = true;

  // invalidate safety flags dependent on ghosting
  validparticleneighbors_ = false;
  validglobalidtolocalindex_ = false;
  validdirectghosting_ = false;
}

void Particle::ParticleEngine::remove_particles_from_containers(
    std::vector<std::set<int>>& particlestoremove)
{
  // iterate over particle types
  for (const auto& type : particlecontainerbundle_->get_particle_types())
  {
    // check for particles of current type
    if (particlestoremove[type].empty()) continue;

    // get container of owned particles of current particle type
    ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

    // iterate in reversed order over particles to be removed
    std::set<int>::reverse_iterator rit;
    for (rit = particlestoremove[type].rbegin(); rit != particlestoremove[type].rend(); ++rit)
      container->remove_particle(*rit);
  }

  // clear after all particles are removed
  particlestoremove.clear();

  // invalidate particle safety flags
  invalidate_particle_safety_flags();
}

void Particle::ParticleEngine::store_positions_after_particle_transfer()
{
  // iterate over particle types
  for (const auto& type : particlecontainerbundle_->get_particle_types())
  {
    // get container of owned particles of current particle type
    ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

    // get number of particles stored in container
    const int particlestored = container->particles_stored();

    // no owned particles of current particle type
    if (particlestored == 0) continue;

    // get pointer to particle states
    const double* pos = container->get_ptr_to_state(Position, 0);
    double* lasttransferpos = container->get_ptr_to_state(LastTransferPosition, 0);

    // get particle state dimension
    int statedim = container->get_state_dim(Position);

    // copy particle position data
    for (int i = 0; i < (statedim * particlestored); ++i) lasttransferpos[i] = pos[i];
  }
}

void Particle::ParticleEngine::relate_owned_particles_to_bins()
{
  // clear vector relating (owned and ghosted) particles to col bins
  particlestobins_.resize(binning_->bincolmap_->num_my_elements());
  for (auto& binIt : particlestobins_) binIt.clear();

  // invalidate particle safety flags
  invalidate_particle_safety_flags();

  // iterate over particle types
  for (const auto& type : particlecontainerbundle_->get_particle_types())
  {
    // get container of owned particles of current particle type
    ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

    // get number of particles stored in container
    const int particlestored = container->particles_stored();

    // no owned particles of current particle type
    if (particlestored <= 0) continue;

    // get pointer to position of particle after last transfer
    const double* lasttransferpos = container->get_ptr_to_state(LastTransferPosition, 0);

    // get particle state dimension
    int statedim = container->get_state_dim(Position);

    // loop over particles in container
    for (int index = 0; index < particlestored; ++index)
    {
      // get global id of bin
      const int gidofbin =
          binning_->binstrategy_->convert_pos_to_gid(&(lasttransferpos[statedim * index]));

#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (gidofbin < 0)
        FOUR_C_THROW("particle out of bounding box but not removed from container!");

      if (binning_->binrowmap_->lid(gidofbin) < 0)
        FOUR_C_THROW("particle not owned by this proc but not removed from container!");
#endif

      // add index relating (owned and ghosted) particles to col bins
      particlestobins_[binning_->bincolmap_->lid(gidofbin)].push_back(std::make_pair(type, index));
    }
  }

  // validate flag denoting valid relation of owned particles to bins
  validownedparticles_ = true;
}

void Particle::ParticleEngine::determine_min_relevant_bin_size()
{
  // get number of bins in all spatial directions
  const auto binperdir = binning_->binstrategy_->bin_per_dir();

  // get bin size
  const std::array<double, 3> binsize = binning_->binstrategy_->bin_size();

  // initialize minimum bin size to maximum bin size
  binning_->minbinsize_ = binning_->binstrategy_->get_max_bin_size();

  // check for minimum bin size in spatial directions with more than one bin layer
  for (int i = 0; i < 3; ++i)
    if (binperdir[i] > 1) binning_->minbinsize_ = std::min(binning_->minbinsize_, binsize[i]);
}

void Particle::ParticleEngine::determine_bin_weights()
{
  // safety check
  if (not validownedparticles_) FOUR_C_THROW("invalid relation of owned particles to bins!");

  // per-pair contribution weight. Pair-list iteration dominates the SPH compute
  // kernels (ComputeDensity, momentum, etc.), so adding a pair-count term to the bin
  // weights gives a much better predictor of actual work than a particle-count-only
  // metric. The factor scales pair cost relative to per-particle setup cost; 1.0 is
  // a reasonable default since both kernel families have comparable per-element cost.
  // Tune here if needed; promoting to an input parameter is a follow-up.
  constexpr double pair_weight_factor = 1.0;

  const int nrowbins = binning_->binrowmap_->num_my_elements();

  // initialize weights of all bins
  binning_->binweights_->put_scalar(1.0e-05);

  double* const weights = binning_->binweights_->get_vector(0).get_values();

  // particle-count contribution (per-particle type-weighted setup cost)
  for (int rowlidofbin = 0; rowlidofbin < nrowbins; ++rowlidofbin)
  {
    // get global id of bin
    const int gidofbin = binning_->binrowmap_->gid(rowlidofbin);

    // iterate over owned particles in current bin
    for (const auto& particleIt : particlestobins_[binning_->bincolmap_->lid(gidofbin)])
    {
      // add weight of particle of specific type
      weights[rowlidofbin] += typeweights_[particleIt.first];
    }
  }

  // pair-count contribution: attribute each pair (i,j) to bin-of-i with weight 2.
  // The factor 2 reflects that the SPH kernels iterate over the pair list once but
  // accumulate symmetrically into both i and j; both half-contributions are work done
  // on this rank and charged here to bin-of-i (a conservative but well-correlated proxy
  // that avoids the extra reverse lookup for ghosted j). Only applied when the pair
  // list is currently valid; otherwise the particle-count fallback above is used.
  if (validparticleneighbors_)
  {
    // build owned-particle (type, ownedindex) -> rowlid lookup from the row-bin entries
    // of particlestobins_. Row bins contain only owned particles by construction.
    std::vector<std::unordered_map<int, int>> owned_to_rowlid(typevectorsize_);
    for (int rowlidofbin = 0; rowlidofbin < nrowbins; ++rowlidofbin)
    {
      const int collid = binning_->bincolmap_->lid(binning_->binrowmap_->gid(rowlidofbin));
      for (const auto& [type, idx] : particlestobins_[collid])
        owned_to_rowlid[type][idx] = rowlidofbin;
    }

    const double per_pair = 2.0 * pair_weight_factor;
    for (const auto& pair : potentialparticleneighbors_)
    {
      // i is always owned by construction of build_particle_to_particle_neighbors
      const auto& [itype, istatus, iindex] = pair.first;
      auto mit = owned_to_rowlid[itype].find(iindex);
      if (mit != owned_to_rowlid[itype].end()) weights[mit->second] += per_pair;
    }
  }

  // R5: neighbor-search cost contribution. last_search_cost_per_rowbin_ holds the
  // measured candidate-pair iteration count per row bin from the most recent
  // build_particle_to_particle_neighbors call. This balances the neighbor-search
  // kernel itself (the dominant cost inside update_connectivity, ~85 s mean / 23 s
  // slack post-R4a) which is driven by pre-cull candidate counts rather than the
  // post-cull pair counts captured by the R4a term above. The two terms are summed
  // directly: per-iteration cost is comparable across both kernel families since
  // both perform a position read plus a distance/kernel evaluation per iteration.
  // The vector is gated on validparticleneighbors_ since the row-bin map and bin
  // contents could otherwise be stale relative to the recorded costs.
  if (validparticleneighbors_ &&
      last_search_cost_per_rowbin_.size() == static_cast<std::size_t>(nrowbins))
  {
    constexpr double search_weight_factor = 1.0;
    for (int rowlidofbin = 0; rowlidofbin < nrowbins; ++rowlidofbin)
      weights[rowlidofbin] += search_weight_factor * last_search_cost_per_rowbin_[rowlidofbin];
  }
}

void Particle::ParticleEngine::invalidate_particle_safety_flags()
{
  validownedparticles_ = false;
  validghostedparticles_ = false;
  validparticleneighbors_ = false;
  validglobalidtolocalindex_ = false;
  validdirectghosting_ = false;
}

FOUR_C_NAMESPACE_CLOSE
