// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_algorithm.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_pstream.hpp"
#include "4C_particle_algorithm_constraints.hpp"
#include "4C_particle_algorithm_gravity.hpp"
#include "4C_particle_algorithm_initial_field.hpp"
#include "4C_particle_algorithm_input_generator.hpp"
#include "4C_particle_algorithm_result_test.hpp"
#include "4C_particle_algorithm_timint.hpp"
#include "4C_particle_algorithm_utils.hpp"
#include "4C_particle_algorithm_viscous_damping.hpp"
#include "4C_particle_engine.hpp"
#include "4C_particle_engine_communication_utils.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_object.hpp"
#include "4C_particle_input.hpp"
#include "4C_particle_interaction_base.hpp"
#include "4C_particle_interaction_dem.hpp"
#include "4C_particle_interaction_sph.hpp"
#include "4C_particle_rigidbody.hpp"
#include "4C_particle_rigidbody_result_test.hpp"
#include "4C_particle_wall.hpp"
#include "4C_particle_wall_result_test.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_result_test.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <cstddef>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::ParticleAlgorithm::ParticleAlgorithm(MPI_Comm comm, const Teuchos::ParameterList& params)
    : AlgorithmBase(*Global::Problem::instance(), comm, params),
      myrank_(Core::Communication::my_mpi_rank(comm)),
      params_(params),
      numparticlesafterlastloadbalance_(0),
      transferevery_(params_.get<bool>("TRANSFER_EVERY")),
      writeresultsevery_(params.get<int>("RESULTSEVERY")),
      writerestartevery_(params.get<int>("RESTARTEVERY")),
      writeresultsthisstep_(true),
      writerestartthisstep_(false),
      isrestarted_(false)
{
  // empty constructor
}

Particle::ParticleAlgorithm::~ParticleAlgorithm() = default;

void Particle::ParticleAlgorithm::init(std::vector<Particle::ParticleObjShrdPtr>& initialparticles)
{
  // init particle engine
  init_particle_engine();

  // init particle wall handler
  init_particle_wall();

  // init rigid body handler
  init_particle_rigid_body();

  // init particle time integration
  init_particle_time_integration();

  // init particle interaction handler
  init_particle_interaction();

  // init particle gravity handler
  init_particle_gravity();

  // init viscous damping handler
  init_viscous_damping();

  // set initial particles to vector of particles to be distributed
  particlestodistribute_ = initialparticles;

  // clear vector of initial particles in global problem
  initialparticles.clear();
}

void Particle::ParticleAlgorithm::setup()
{
  // generate initial particles
  if (not isrestarted_) generate_initial_particles();

  // determine all particle types
  determine_particle_types();

  // determine particle states of all particle types
  determine_particle_states_of_particle_types();

  // setup particle engine
  particleengine_->setup(particlestatestotypes_);

  // setup wall handler
  if (particlewall_) particlewall_->setup(particleengine_, time());

  // setup rigid body handler
  if (particlerigidbody_) particlerigidbody_->setup(particleengine_);

  // setup particle time integration
  particletimint_->setup(
      particleengine_, particlerigidbody_, create_constraints(params_, get_comm()));

  // setup particle interaction handler
  if (particleinteraction_) particleinteraction_->setup(particleengine_, particlewall_);

  // setup viscous damping handler
  if (viscousdamping_) viscousdamping_->setup(particleengine_);

  // setup initial particles
  setup_initial_particles();

  // build Dirichlet BC function cache after particles are distributed to containers
  particletimint_->build_dirichlet_bc_funct_cache(get_comm());

  // setup initial rigid bodies
  if (particlerigidbody_) setup_initial_rigid_bodies();

  // distribute load among processors (first pass: by particle count)
  distribute_load_among_procs();

  // ghost particles on other processors
  particleengine_->ghost_particles();

  // build global id to local index map
  particleengine_->build_global_id_to_local_index_map();

  // build potential neighbor relation — populates bin_interaction_costs_ for cost-aware rebalancing
  if (particleinteraction_) build_potential_neighbor_relation();

  // second redistribution pass: redistribute load using interaction-pair counts as bin weights
  // now that bin_interaction_costs_ is available from the neighbor build above
  if (particleinteraction_)
  {
    distribute_load_among_procs();

    // re-ghost and rebuild after second redistribution
    particleengine_->ghost_particles();
    particleengine_->build_global_id_to_local_index_map();
    build_potential_neighbor_relation();
  }

  // print per-type interaction statistics (measured load-balancing weights for this geometry)
  if (particleinteraction_) particleengine_->print_particle_interaction_statistics();

  // setup initial states
  if (not isrestarted_) setup_initial_states();

  // write initial output
  if (not isrestarted_) write_output();
}

void Particle::ParticleAlgorithm::read_restart(const int restartstep)
{
  // clear vector of particles to be distributed
  particlestodistribute_.clear();

  // create discretization reader
  const std::shared_ptr<Core::IO::DiscretizationReader> reader =
      particleengine_->bin_dis_reader(restartstep);

  // safety check
  if (restartstep != reader->read_int("step"))
    FOUR_C_THROW("time step on file not equal to given step!");

  // get restart time
  double restarttime = reader->read_double("time");

  // read restart of particle engine
  particleengine_->read_restart(reader, particlestodistribute_);

  // read restart of rigid body handler
  if (particlerigidbody_) particlerigidbody_->read_restart(reader);

  // read restart of particle interaction handler
  if (particleinteraction_) particleinteraction_->read_restart(reader);

  // read restart of wall handler
  if (particlewall_) particlewall_->read_restart(restartstep);

  // set time and step after restart
  set_time_step(restarttime, restartstep);

  // set flag indicating restart to true
  isrestarted_ = true;

  // short screen output
  if (myrank_ == 0)
    Core::IO::cout << "====== restart of the particle simulation from step " << restartstep
                   << Core::IO::endl;
}

void Particle::ParticleAlgorithm::timeloop()
{
  // time loop
  while (not_finished())
  {
    // counter and print header
    prepare_time_step();

    // pre evaluate time step
    pre_evaluate_time_step();

    // integrate time step
    integrate_time_step();

    // post evaluate time step
    post_evaluate_time_step();

    // write output
    write_output();

    // write restart information
    write_restart();
  }
}

void Particle::ParticleAlgorithm::prepare_time_step(bool do_print_header)
{
  // increment time and step
  increment_time_and_step();

  // set current time
  set_current_time();

  // set current step size
  set_current_step_size();

  // print header
  if (do_print_header) print_header();

  // update result and restart control flags
  writeresultsthisstep_ = (writeresultsevery_ and (step() % writeresultsevery_ == 0));
  writerestartthisstep_ = (writerestartevery_ and (step() % writerestartevery_ == 0));

  // set current write result flag
  set_current_write_result_flag();
}

void Particle::ParticleAlgorithm::pre_evaluate_time_step()
{
  // pre evaluate time step
  if (particleinteraction_) particleinteraction_->pre_evaluate_time_step();
}

void Particle::ParticleAlgorithm::integrate_time_step()
{
  // time integration scheme specific pre-interaction routine
  particletimint_->pre_interaction_routine();

  // update connectivity
  update_connectivity();

  // evaluate time step
  evaluate_time_step();

  // time integration scheme specific post-interaction routine
  particletimint_->post_interaction_routine();
}

void Particle::ParticleAlgorithm::post_evaluate_time_step()
{
  // post evaluate time step
  std::vector<Particle::ParticleTypeToType> particlesfromphasetophase;
  if (particleinteraction_)
    particleinteraction_->post_evaluate_time_step(particlesfromphasetophase);

  if (particlerigidbody_)
  {
    // have rigid body phase change
    if (particlerigidbody_->have_rigid_body_phase_change(particlesfromphasetophase))
    {
      // update connectivity
      update_connectivity();

      // evaluate rigid body phase change
      particlerigidbody_->evaluate_rigid_body_phase_change(particlesfromphasetophase);
    }
  }
}

void Particle::ParticleAlgorithm::write_output() const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::ParticleAlgorithm::WriteOutput");

  // write result step
  if (writeresultsthisstep_)
  {
    // write particle runtime output
    particleengine_->write_particle_runtime_output(step(), time());

    // write binning discretization output (debug feature)
    particleengine_->write_bin_dis_output(step(), time());

    // write rigid body runtime output
    if (particlerigidbody_) particlerigidbody_->write_rigid_body_runtime_output(step(), time());

    // write interaction runtime output
    if (particleinteraction_)
      particleinteraction_->write_interaction_runtime_output(step(), time());

    // write wall runtime output
    if (particlewall_) particlewall_->write_wall_runtime_output(step(), time());
  }
}

void Particle::ParticleAlgorithm::write_restart() const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::ParticleAlgorithm::write_restart");

  // write restart step
  if (writerestartthisstep_)
  {
    // write restart of particle engine
    particleengine_->write_restart(step(), time());

    // write restart of rigid body handler
    if (particlerigidbody_) particlerigidbody_->write_restart();

    // write restart of particle interaction handler
    if (particleinteraction_) particleinteraction_->write_restart();

    // write restart of wall handler
    if (particlewall_) particlewall_->write_restart(step(), time());

    // short screen output
    if (myrank_ == 0)
      Core::IO::cout(Core::IO::verbose)
          << "====== restart of the particle simulation written in step " << step()
          << Core::IO::endl;
  }
}

std::vector<std::shared_ptr<Core::Utils::ResultTest>>
Particle::ParticleAlgorithm::create_result_tests()
{
  // build global id to local index map
  particleengine_->build_global_id_to_local_index_map();

  // particle field specific result test objects
  std::vector<std::shared_ptr<Core::Utils::ResultTest>> allresulttests;

  // particle result test
  {
    // create particle result test
    std::shared_ptr<Particle::ParticleResultTest> particleresulttest =
        std::make_shared<Particle::ParticleResultTest>();

    // setup particle result test
    particleresulttest->setup(particleengine_);

    allresulttests.push_back(particleresulttest);
  }

  // wall result test
  if (particlewall_)
  {
    // create wall result test
    std::shared_ptr<Particle::WallResultTest> wallresulttest =
        std::make_shared<Particle::WallResultTest>();

    // setup wall result test
    wallresulttest->setup(particlewall_);

    allresulttests.push_back(wallresulttest);
  }

  if (particlerigidbody_)
  {
    // create rigid body result test
    std::shared_ptr<Particle::RigidBodyResultTest> rigidbodyresulttest =
        std::make_shared<Particle::RigidBodyResultTest>();

    // setup rigid body result test
    rigidbodyresulttest->setup(particlerigidbody_);

    allresulttests.push_back(rigidbodyresulttest);
  }

  return allresulttests;
}

void Particle::ParticleAlgorithm::init_particle_engine()
{
  // create and init particle engine
  particleengine_ = std::make_shared<Particle::ParticleEngine>(get_comm(), params_);
}

void Particle::ParticleAlgorithm::init_particle_wall()
{
  // get type of particle wall source
  auto particlewallsource =
      Teuchos::getIntegralValue<Particle::ParticleWallSource>(params_, "PARTICLE_WALL_SOURCE");

  // create particle wall handler
  switch (particlewallsource)
  {
    case Particle::NoParticleWall:
    {
      particlewall_ = std::shared_ptr<Particle::WallHandlerBase>(nullptr);
      break;
    }
    case Particle::DiscretCondition:
    {
      particlewall_ = std::make_shared<Particle::WallHandlerDiscretCondition>(get_comm(), params_);
      break;
    }
    case Particle::BoundingBox:
    {
      particlewall_ = std::make_shared<Particle::WallHandlerBoundingBox>(get_comm(), params_);
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown type of particle wall source!");
      break;
    }
  }

  // init particle wall handler
  if (particlewall_) particlewall_->init(particleengine_->get_binning_strategy());
}

void Particle::ParticleAlgorithm::init_particle_rigid_body()
{
  // create rigid body handler
  if (params_.get<bool>("RIGID_BODY_MOTION"))
    particlerigidbody_ = std::make_shared<Particle::RigidBodyHandler>(get_comm(), params_);
}

void Particle::ParticleAlgorithm::init_particle_time_integration()
{
  // get particle time integration scheme
  auto timinttype = Teuchos::getIntegralValue<Particle::DynamicType>(params_, "DYNAMICTYPE");

  // create particle time integration
  switch (timinttype)
  {
    case Particle::dyna_semiimpliciteuler:
    {
      particletimint_ = std::unique_ptr<Particle::TimIntSemiImplicitEuler>(
          new Particle::TimIntSemiImplicitEuler(params_));
      break;
    }
    case Particle::dyna_velocityverlet:
    {
      particletimint_ = std::unique_ptr<Particle::TimIntVelocityVerlet>(
          new Particle::TimIntVelocityVerlet(params_));
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown particle time integration scheme!");
      break;
    }
  }
}

void Particle::ParticleAlgorithm::init_particle_interaction()
{
  // get particle interaction type
  auto interactiontype =
      Teuchos::getIntegralValue<Particle::InteractionType>(params_, "INTERACTION");

  // create particle interaction handler
  switch (interactiontype)
  {
    case Particle::interaction_none:
    {
      particleinteraction_ = std::unique_ptr<Particle::ParticleInteractionBase>(nullptr);
      break;
    }
    case Particle::interaction_sph:
    {
      particleinteraction_ = std::unique_ptr<Particle::ParticleInteractionSPH>(
          new Particle::ParticleInteractionSPH(get_comm(), params_));
      break;
    }
    case Particle::interaction_dem:
    {
      particleinteraction_ = std::unique_ptr<Particle::ParticleInteractionDEM>(
          new Particle::ParticleInteractionDEM(get_comm(), params_));
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown particle interaction type!");
      break;
    }
  }
}

void Particle::ParticleAlgorithm::init_particle_gravity()
{
  // init gravity acceleration vector
  std::vector<double> gravity;
  std::string value;
  std::istringstream gravitystream(
      Teuchos::getNumericStringParameter(params_, "GRAVITY_ACCELERATION"));

  while (gravitystream >> value) gravity.push_back(std::atof(value.c_str()));

  // safety check
  if (static_cast<int>(gravity.size()) != 3)
    FOUR_C_THROW("dimension (dim = {}) of gravity acceleration vector is wrong!",
        static_cast<int>(gravity.size()));

  // get magnitude of gravity
  double temp = 0.0;
  for (double g : gravity) temp += g * g;
  const double gravity_norm = std::sqrt(temp);

  // create particle gravity handler
  if (gravity_norm > 0.0)
    particlegravity_ =
        std::make_unique<Particle::GravityHandler>(gravity, params_.get<int>("GRAVITY_RAMP_FUNCT"));
}

void Particle::ParticleAlgorithm::init_viscous_damping()
{
  // get viscous damping factor
  const double viscdampfac = params_.get<double>("VISCOUS_DAMPING");

  // create viscous damping handler
  if (viscdampfac > 0.0)
    viscousdamping_ = std::unique_ptr<Particle::ViscousDampingHandler>(
        new Particle::ViscousDampingHandler(viscdampfac));
}

void Particle::ParticleAlgorithm::generate_initial_particles()
{
  // create particle input generator
  std::unique_ptr<Particle::InputGenerator> particleinputgenerator =
      std::make_unique<Particle::InputGenerator>(get_comm(), params_);

  // generate particles
  particleinputgenerator->generate_particles(particlestodistribute_);
}

void Particle::ParticleAlgorithm::determine_particle_types()
{
  // init map relating particle types to dynamic load balance factor
  std::map<Particle::TypeEnum, double> typetodynloadbal;

  // read parameters relating particle types to values
  ParticleUtils::read_params_types_related_to_values(
      params_, "PHASE_TO_DYNLOADBALFAC", typetodynloadbal);

  // insert into map of particle types and corresponding states with empty set
  for (auto& typeIt : typetodynloadbal)
    particlestatestotypes_.insert(std::make_pair(typeIt.first, std::set<Particle::StateEnum>()));

  // safety check
  for (auto& particle : particlestodistribute_)
    if (not particlestatestotypes_.contains(particle->return_particle_type()))
      FOUR_C_THROW("particle type '{}' of initial particle not defined!",
          Particle::enum_to_type_name(particle->return_particle_type()));
}

void Particle::ParticleAlgorithm::determine_particle_states_of_particle_types()
{
  // iterate over particle types
  for (auto& typeIt : particlestatestotypes_)
  {
    // set of particle states for current particle type
    std::set<Particle::StateEnum>& particlestates = typeIt.second;

    // insert default particle states
    particlestates.insert({Particle::Position, Particle::Velocity, Particle::Acceleration,
        Particle::LastTransferPosition});
  }

  // insert integration dependent states of all particle types
  particletimint_->insert_particle_states_of_particle_types(particlestatestotypes_);

  // insert interaction dependent states of all particle types
  if (particleinteraction_)
    particleinteraction_->insert_particle_states_of_particle_types(particlestatestotypes_);

  // insert wall handler dependent states of all particle types
  if (particlewall_)
    particlewall_->insert_particle_states_of_particle_types(particlestatestotypes_);

  // insert rigid body handler dependent states of all particle types
  if (particlerigidbody_)
    particlerigidbody_->insert_particle_states_of_particle_types(particlestatestotypes_);
}

void Particle::ParticleAlgorithm::setup_initial_particles()
{
  // get unique global ids for all particles
  if (not isrestarted_)
    particleengine_->get_unique_global_ids_for_all_particles(particlestodistribute_);

  // erase particles outside bounding box
  particleengine_->erase_particles_outside_bounding_box(particlestodistribute_);

  // distribute particles to owning processor
  particleengine_->distribute_particles(particlestodistribute_);

  // distribute interaction history
  if (particleinteraction_) particleinteraction_->distribute_interaction_history();
}

void Particle::ParticleAlgorithm::setup_initial_rigid_bodies()
{
  // set initial affiliation pair data
  if (not isrestarted_) particlerigidbody_->set_initial_affiliation_pair_data();

  // set unique global ids for all rigid bodies
  if (not isrestarted_) particlerigidbody_->set_unique_global_ids_for_all_rigid_bodies();

  // allocate rigid body states
  if (not isrestarted_) particlerigidbody_->allocate_rigid_body_states();

  // distribute rigid body
  particlerigidbody_->distribute_rigid_body();
}

void Particle::ParticleAlgorithm::setup_initial_states()
{
  // set initial states
  if (particleinteraction_) particleinteraction_->set_initial_states();

  // initialize rigid body mass quantities and orientation
  if (particlerigidbody_)
    particlerigidbody_->initialize_rigid_body_mass_quantities_and_orientation();

  // set initial conditions
  set_initial_conditions();

  // time integration scheme specific initialization routine
  particletimint_->set_initial_states();

  // evaluate consistent initial states
  {
    // pre evaluate time step
    pre_evaluate_time_step();

    // update connectivity
    update_connectivity();

    // evaluate time step
    evaluate_time_step();

    // post evaluate time step
    post_evaluate_time_step();
  }
}

void Particle::ParticleAlgorithm::update_connectivity()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::ParticleAlgorithm::update_connectivity");

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // check number of unique global ids
  particleengine_->check_number_of_unique_global_ids();
#endif

  // check particle interaction distance concerning bin size
  if (particleinteraction_)
    particleinteraction_->check_particle_interaction_distance_concerning_bin_size();

  // check that wall nodes are located in bounding box
  if (particlewall_) particlewall_->check_wall_nodes_located_in_bounding_box();

  if (check_load_transfer_needed())
  {
    // transfer load between processors
    transfer_load_between_procs();

    // distribute load among processors
    if (check_load_redistribution_needed()) distribute_load_among_procs();

    // ghost particles on other processors
    particleengine_->ghost_particles();

    // build global id to local index map
    particleengine_->build_global_id_to_local_index_map();

    // build potential neighbor relation
    if (particleinteraction_) build_potential_neighbor_relation();
  }
  else
  {
    // refresh particles being ghosted on other processors
    particleengine_->refresh_particles();
  }
}

bool Particle::ParticleAlgorithm::check_load_transfer_needed()
{
  bool transferload = transferevery_ or writeresultsthisstep_ or writerestartthisstep_;

  // check max position increment
  transferload |= check_max_position_increment();

  // check for valid particle connectivity
  transferload |= (not particleengine_->have_valid_particle_connectivity());

  // check for valid particle neighbors
  if (particleinteraction_) transferload |= (not particleengine_->have_valid_particle_neighbors());

  // check for valid wall neighbors
  if (particleinteraction_ and particlewall_)
    transferload |= (not particlewall_->have_valid_wall_neighbors());

  return transferload;
}

bool Particle::ParticleAlgorithm::check_max_position_increment()
{
  // get maximum particle interaction distance
  double allprocmaxinteractiondistance = 0.0;
  if (particleinteraction_)
  {
    double maxinteractiondistance = particleinteraction_->max_interaction_distance();
    allprocmaxinteractiondistance =
        Core::Communication::max_all(maxinteractiondistance, get_comm());
  }

  // get max particle position increment since last transfer
  double maxparticlepositionincrement = get_max_particle_position_increment();

  // get max wall position increment since last transfer
  double maxwallpositionincrement = 0.0;
  if (particlewall_) particlewall_->get_max_wall_position_increment(maxwallpositionincrement);

  // get max overall position increment since last transfer
  const double maxpositionincrement =
      std::max(maxparticlepositionincrement, maxwallpositionincrement);

  // get allowed position increment
  const double allowedpositionincrement =
      0.5 * (particleengine_->min_bin_size() - allprocmaxinteractiondistance);

  // check if a particle transfer is needed based on a worst case scenario:
  // two particles approach each other with maximum position increment in one spatial dimension
  return (maxpositionincrement > allowedpositionincrement);
}

double Particle::ParticleAlgorithm::get_max_particle_position_increment()
{
  // Struct to collect additional debug information to throw a more informative error message
  struct DebugOutput
  {
    std::map<ParticleType, std::size_t>
        violating_particles_per_type;  //! Map of particle type to number of particles violating the
                                       //! condition (maxpositionincrement < minimum_bin_size)
    int lid_of_max_position_increment;  //! local id of the particle with the maximum position
                                        //! increment
    ParticleType particle_type_of_max_position_increment;  //! the particle type of the particle
                                                           //! with the maximum position increment
  };

  const double minimum_bin_size = particleengine_->min_bin_size();

  // Lambda function to find the maximum position increment. In case the maximum position increment
  // is larger than the minimum bin size, the function is reused to output additional debug
  // information. The performance does not matter because we exit the simulation anyway. For the
  // normal calculation DebugOutput is not set such that the compiler can optimize away the
  // additional if branches so that we do not have a performance overhead.
  // The lambda function is chosen over a separate private function in order to have everything in
  // one place
  auto find_max_position_increment = [&](DebugOutput* debug_output = nullptr) -> double
  {
    // maximum position increment since last particle transfer
    double maxpositionincrement = 0.0;

    // get particle container bundle
    Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
        particleengine_->get_particle_container_bundle();

    // iterate over particle types
    for (const auto& typeEnum : particlecontainerbundle->get_particle_types())
    {
      if (debug_output)
      {
        debug_output->violating_particles_per_type[typeEnum] = 0;
      }
      // get container of owned particles of current particle type
      Particle::ParticleContainer* container =
          particlecontainerbundle->get_specific_container(typeEnum, Particle::Owned);

      // get number of particles stored in container
      const int particlestored = container->particles_stored();

      // no owned particles of current particle type
      if (particlestored == 0) continue;

      // get particle state dimension
      int statedim = container->get_state_dim(Particle::Position);

      // position increment of particle
      double positionincrement[3];

      // iterate over owned particles of current type
      for (int i = 0; i < particlestored; ++i)
      {
        // get pointer to particle states
        const double* pos = container->get_ptr_to_state(Particle::Position, i);
        const double* lasttransferpos =
            container->get_ptr_to_state(Particle::LastTransferPosition, i);

        // position increment of particle considering periodic boundaries
        particleengine_->distance_between_particles(pos, lasttransferpos, positionincrement);


        if (debug_output)
        {
          double max_position_increment_of_particle = 0.0;
          for (int dim = 0; dim < statedim; ++dim)
          {
            max_position_increment_of_particle =
                std::max(max_position_increment_of_particle, std::abs(positionincrement[dim]));
          }
          if (max_position_increment_of_particle > minimum_bin_size)
          {
            debug_output->violating_particles_per_type[typeEnum]++;
            // Save gid and type of particle with maximum position increment
            if (max_position_increment_of_particle > maxpositionincrement)
            {
              debug_output->lid_of_max_position_increment = i;
              debug_output->particle_type_of_max_position_increment = typeEnum;
            }
          }
        }

        // iterate over spatial dimension to update maximum position increment
        for (int dim = 0; dim < statedim; ++dim)
        {
          maxpositionincrement = std::max(maxpositionincrement, std::abs(positionincrement[dim]));
        }
      }
    }
    return maxpositionincrement;
  };

  double maxpositionincrement = find_max_position_increment();

  // bin size safety check
  if (maxpositionincrement > minimum_bin_size)
  {
    // If we end up here we get additional information and throw an error
    auto debug_output = std::make_unique<DebugOutput>();
    maxpositionincrement = find_max_position_increment(debug_output.get());

    std::stringstream ss;
    std::size_t total_number_of_violating_particles = 0;
    for (const auto& [key, value] : debug_output->violating_particles_per_type)
    {
      if (value == 0) continue;

      total_number_of_violating_particles += value;
      ss << "    " << enum_to_type_name(key) << ": " << value << "\n";
    }

    // Get position and velocity of particle with maximum position increment
    Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
        particleengine_->get_particle_container_bundle();
    ParticleContainer* container = particlecontainerbundle->get_specific_container(
        debug_output->particle_type_of_max_position_increment, Owned);
    const double* position =
        container->get_ptr_to_state(Position, debug_output->lid_of_max_position_increment);
    const double* velocity =
        container->get_ptr_to_state(Velocity, debug_output->lid_of_max_position_increment);
    const int gid = *container->get_ptr_to_global_id(debug_output->lid_of_max_position_increment);
    FOUR_C_THROW(
        "{} particle(s) traveled more than one bin on this processor.\n"
        "  Minimum bin size: {}\n"
        "  Maximum position increment: {} for particle with\n"
        "    gid: {} of type {}\n"
        "    Position: ({}, {}, {})\n"
        "    Velocity: ({}, {}, {})\n"
        "  Particle type: number of violating particles\n"
        "{}"
        "This can have numerous causes and can hint at an unstable simulation. "
        "Carefully check the time step size and other interaction related parameters.",
        total_number_of_violating_particles, minimum_bin_size, maxpositionincrement, gid,
        enum_to_type_name(debug_output->particle_type_of_max_position_increment), position[0],
        position[1], position[2], velocity[0], velocity[1], velocity[2], ss.str());
  }

  // get maximum particle position increment on all processors
  double allprocmaxpositionincrement =
      Core::Communication::max_all(maxpositionincrement, get_comm());

  return allprocmaxpositionincrement;
}

void Particle::ParticleAlgorithm::transfer_load_between_procs()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::ParticleAlgorithm::transfer_load_between_procs");

  // transfer particles to new bins and processors
  particleengine_->transfer_particles();

  // transfer wall elements and nodes
  if (particlewall_) particlewall_->transfer_wall_elements_and_nodes();

  // communicate rigid body
  if (particlerigidbody_) particlerigidbody_->communicate_rigid_body();

  // communicate interaction history
  if (particleinteraction_) particleinteraction_->communicate_interaction_history();

  // short screen output
  if (myrank_ == 0)
    Core::IO::cout(Core::IO::verbose) << "transfer load in step " << step() << Core::IO::endl;
}

bool Particle::ParticleAlgorithm::check_load_redistribution_needed()
{
  bool redistributeload = writerestartthisstep_;

  // percentage limit
  const double percentagelimit = 0.1;

  // get number of particles on this processor
  int numberofparticles = particleengine_->get_number_of_particles();

  // percentage change of particles on this processor
  double percentagechange = 0.0;
  if (numparticlesafterlastloadbalance_ > 0)
    percentagechange =
        std::abs(static_cast<double>(numberofparticles - numparticlesafterlastloadbalance_) /
                 numparticlesafterlastloadbalance_);

  // get maximum percentage change of particles
  double maxpercentagechange = 0.0;
  maxpercentagechange = Core::Communication::max_all(percentagechange, get_comm());

  // criterion for load redistribution based on maximum percentage change of the number of particles
  redistributeload |= (maxpercentagechange > percentagelimit);

  return redistributeload;
}

void Particle::ParticleAlgorithm::distribute_load_among_procs()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::ParticleAlgorithm::distribute_load_among_procs");

  // dynamic load balancing
  particleengine_->dynamic_load_balancing();

  // get number of particles on this processor
  numparticlesafterlastloadbalance_ = particleengine_->get_number_of_particles();

  if (particlewall_)
  {
    // update bin row and column map
    particlewall_->update_bin_row_and_col_map(
        particleengine_->get_bin_row_map(), particleengine_->get_bin_col_map());

    // distribute wall elements and nodes
    particlewall_->distribute_wall_elements_and_nodes();
  }

  // communicate rigid body
  if (particlerigidbody_) particlerigidbody_->communicate_rigid_body();

  // communicate interaction history
  if (particleinteraction_) particleinteraction_->communicate_interaction_history();

  // short screen output
  if (myrank_ == 0)
    Core::IO::cout(Core::IO::verbose) << "distribute load in step " << step() << Core::IO::endl;
}

void Particle::ParticleAlgorithm::build_potential_neighbor_relation()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::ParticleAlgorithm::build_potential_neighbor_relation");

  // build particle to particle neighbors
  particleengine_->build_particle_to_particle_neighbors();

  if (particlewall_)
  {
    // relate bins to column wall elements
    particlewall_->relate_bins_to_col_wall_eles();

    // build particle to wall neighbors
    particlewall_->build_particle_to_wall_neighbors(particleengine_->get_particles_to_bins());
  }
}

void Particle::ParticleAlgorithm::set_initial_conditions()
{
  // create and init particle initial field handler
  std::unique_ptr<Particle::InitialFieldHandler> initialfield =
      std::make_unique<Particle::InitialFieldHandler>(params_);

  // setup particle initial field handler
  initialfield->setup(particleengine_);

  // set initial fields
  initialfield->set_initial_fields();

  // set rigid body initial conditions
  if (particlerigidbody_) particlerigidbody_->set_initial_conditions();
}

void Particle::ParticleAlgorithm::set_current_time()
{
  // set current time in particle time integration
  particletimint_->set_current_time(time());

  // set current time in particle interaction
  if (particleinteraction_) particleinteraction_->set_current_time(time());
}

void Particle::ParticleAlgorithm::set_current_step_size()
{
  // set current step size in particle interaction
  if (particleinteraction_) particleinteraction_->set_current_step_size(dt());
}

void Particle::ParticleAlgorithm::set_current_write_result_flag()
{
  // set current write result flag in particle interaction
  if (particleinteraction_)
    particleinteraction_->set_current_write_result_flag(writeresultsthisstep_);
}

void Particle::ParticleAlgorithm::evaluate_time_step()
{
  // clear forces and torques
  if (particlerigidbody_) particlerigidbody_->clear_forces_and_torques();

  // set gravity acceleration
  if (particlegravity_) set_gravity_acceleration();

  // evaluate particle interactions
  if (particleinteraction_) particleinteraction_->evaluate_interactions();

  // print per-type interaction cost table once, after the first evaluation
  if (particleinteraction_ and not interaction_cost_printed_)
  {
    print_particle_interaction_cost();
    interaction_cost_printed_ = true;
  }

  // apply viscous damping contribution
  if (viscousdamping_) viscousdamping_->apply_viscous_damping();

  // compute accelerations of rigid bodies
  if (particlerigidbody_ and particleinteraction_) particlerigidbody_->compute_accelerations();
}

void Particle::ParticleAlgorithm::set_gravity_acceleration()
{
  std::vector<double> scaled_gravity(3);

  // get gravity acceleration
  particlegravity_->get_gravity_acceleration(time(), scaled_gravity);

  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengine_->get_particle_container_bundle();

  // iterate over particle types
  for (auto& typeEnum : particlecontainerbundle->get_particle_types())
  {
    // gravity is not set for boundary or rigid particles
    if (typeEnum == Particle::BoundaryPhase or typeEnum == Particle::RigidPhase) continue;

    // gravity is not set for open boundary particles
    if (typeEnum == Particle::DirichletPhase or typeEnum == Particle::NeumannPhase) continue;

    // set gravity acceleration for all particles of current type
    particlecontainerbundle->set_state_specific_container(
        scaled_gravity, Particle::Acceleration, typeEnum);
  }

  // add gravity acceleration
  if (particlerigidbody_) particlerigidbody_->add_gravity_acceleration(scaled_gravity);

  // set scaled gravity in particle interaction handler
  if (particleinteraction_) particleinteraction_->set_gravity(scaled_gravity);
}

void Particle::ParticleAlgorithm::print_particle_interaction_cost()
{
  // collect per-type stats from the interaction handler (SPH-specific; no-op for DEM)
  std::map<ParticleType, long> sph_pairs;
  long pd_bond_pairs = 0;
  double sph_time_ns = 0.0;
  double pd_time_ns = 0.0;
  particleinteraction_->collect_interaction_type_stats(
      sph_pairs, pd_bond_pairs, sph_time_ns, pd_time_ns);

  // no stats available (e.g. DEM) → nothing to print
  if (sph_pairs.empty() and pd_bond_pairs == 0) return;

  // ordered list of stored particle types (same order as the engine diagnostic)
  const auto& type_set = particleengine_->get_particle_container_bundle()->get_particle_types();
  std::vector<ParticleType> types(type_set.begin(), type_set.end());

  // global particle counts per type (reduce local counts across all ranks)
  std::vector<long> particle_counts(types.size(), 0);
  for (std::size_t i = 0; i < types.size(); ++i)
    particle_counts[i] =
        static_cast<long>(particleengine_->get_number_of_particles_of_specific_type(types[i]));
  MPI_Allreduce(MPI_IN_PLACE, particle_counts.data(), static_cast<int>(particle_counts.size()),
      MPI_LONG, MPI_SUM, get_comm());

  // global actual SPH pair counts per type
  std::vector<long> sph_pair_counts(types.size(), 0);
  for (std::size_t i = 0; i < types.size(); ++i)
  {
    auto it = sph_pairs.find(types[i]);
    if (it != sph_pairs.end()) sph_pair_counts[i] = it->second;
  }
  MPI_Allreduce(MPI_IN_PLACE, sph_pair_counts.data(), static_cast<int>(sph_pair_counts.size()),
      MPI_LONG, MPI_SUM, get_comm());

  // global PD bond pair count
  MPI_Allreduce(MPI_IN_PLACE, &pd_bond_pairs, 1, MPI_LONG, MPI_SUM, get_comm());

  // average SPH and PD timing over all ranks for a less rank-0-biased estimate
  double times[2] = {sph_time_ns, pd_time_ns};
  MPI_Allreduce(MPI_IN_PLACE, times, 2, MPI_DOUBLE, MPI_SUM, get_comm());
  const int nproc = Core::Communication::num_mpi_ranks(get_comm());
  const double avg_sph_time_ns = times[0] / nproc;
  const double avg_pd_time_ns = times[1] / nproc;

  // cost per interaction [ns/interaction]
  // SPH cost is attributed uniformly to all SPH pair participants
  const long total_sph_pairs = [&]()
  {
    long s = 0;
    for (const long c : sph_pair_counts) s += c;
    // each pair is counted twice (once per endpoint), so divide by 2 to get unique pairs
    return s / 2;
  }();
  // PD bond pairs: each pair involves 2 PD particles → total PD interactions = 2 × bond count
  const long total_pd_interactions = 2 * pd_bond_pairs;

  const double cost_sph = (total_sph_pairs > 0) ? avg_sph_time_ns / total_sph_pairs : 0.0;
  const double cost_pd_bond = (pd_bond_pairs > 0) ? avg_pd_time_ns / pd_bond_pairs : 0.0;

  // only rank 0 prints
  if (myrank_ != 0) return;

  // for each type compute combined cost-weighted load per particle
  // w_i = (SPH interactions/particle) × cost_sph + (PD bond interactions/particle) × cost_pd_bond
  std::vector<double> combined_weight(types.size(), 0.0);
  for (std::size_t i = 0; i < types.size(); ++i)
  {
    if (particle_counts[i] == 0) continue;
    const double sph_ipp =
        static_cast<double>(sph_pair_counts[i]) / static_cast<double>(particle_counts[i]);
    // PD interactions/particle: 2 × pd_bond_pairs / N_pd; only non-zero for the PDPhase type
    const double pd_ipp =
        (types[i] == Particle::PDPhase and particle_counts[i] > 0)
            ? static_cast<double>(total_pd_interactions) / static_cast<double>(particle_counts[i])
            : 0.0;
    combined_weight[i] = sph_ipp * cost_sph + pd_ipp * cost_pd_bond;
  }

  // normalize weights to the type with the smallest non-zero weight
  double ref_weight = 0.0;
  for (const double w : combined_weight)
    if (w > 0.0 and (ref_weight == 0.0 or w < ref_weight)) ref_weight = w;

  std::ostringstream table;
  const std::string ruler =
      "+----------------------+-----------------+------------------+------------------+------------"
      "------"
      "-+\n";
  table << "\n" << ruler;
  table << "| per-type interaction cost estimate (first time-step, rank-averaged timing)           "
           "     "
           "        |\n";
  table << ruler;
  table << "| " << std::left << std::setw(20) << "particle type" << " | " << std::right
        << std::setw(15) << "SPH pairs/p" << " | " << std::setw(16) << "PD bonds/p" << " | "
        << std::setw(16) << "cost_sph [ns/p]" << " | " << std::setw(16) << "cost_pd [ns/p]"
        << " |\n";
  table << ruler;
  for (std::size_t i = 0; i < types.size(); ++i)
  {
    if (particle_counts[i] == 0) continue;
    const double sph_ipp =
        static_cast<double>(sph_pair_counts[i]) / static_cast<double>(particle_counts[i]);
    const double pd_ipp =
        (types[i] == Particle::PDPhase and particle_counts[i] > 0)
            ? static_cast<double>(total_pd_interactions) / static_cast<double>(particle_counts[i])
            : 0.0;
    table << "| " << std::left << std::setw(20) << enum_to_type_name(types[i]) << " | "
          << std::right << std::fixed << std::setprecision(2) << std::setw(15) << sph_ipp << " | "
          << std::setw(16) << pd_ipp << " | " << std::setw(16) << sph_ipp * cost_sph << " | "
          << std::setw(16) << pd_ipp * cost_pd_bond << " |\n";
  }
  table << ruler;
  table << "| SPH: " << total_sph_pairs << " unique pairs, avg cost " << std::fixed
        << std::setprecision(3) << cost_sph << " ns/pair  |  PD bonds: " << pd_bond_pairs
        << ", avg cost " << cost_pd_bond << " ns/bond\n";
  table << ruler;

  // recommended relative weights
  table << "| recommended relative weights (for PHASE_TO_DYNLOADBALFAC or rank sizing):\n";
  for (std::size_t i = 0; i < types.size(); ++i)
  {
    if (particle_counts[i] == 0) continue;
    const double rel = (ref_weight > 0.0) ? combined_weight[i] / ref_weight : 0.0;
    table << "|   " << std::left << std::setw(20) << enum_to_type_name(types[i]) << " = "
          << std::right << std::fixed << std::setprecision(3) << rel << "\n";
  }
  table << ruler;

  Core::IO::cout << table.str() << Core::IO::endl;
}

FOUR_C_NAMESPACE_CLOSE
