#include <iostream>

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include "CLI_mpi.hpp"

#include "AmsSort/AmsSort.hpp"

#include <kamping/collectives/allgather.hpp>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/measurements/counter.hpp>
#include <kamping/measurements/printer.hpp>
#include <random>

struct Parameters {
  std::size_t datasize = 1000;
  std::size_t num_pe = -1;
  std::size_t iterations = 1;
  std::size_t num_levels = 1;
  bool hierarchy_aware = false;
  double imbalance = 1.10;
  bool not_use_dma = false;
  bool not_use_mpi_colls = false;
  std::string output_path;

  auto config() const {
    std::vector<std::pair<std::string, std::string>> config_vector;
    config_vector.emplace_back("datasize", std::to_string(datasize));
    config_vector.emplace_back("num_pe", std::to_string(num_pe));
    config_vector.emplace_back("imbalance", std::to_string(imbalance));
    config_vector.emplace_back("not_use_dma", std::to_string(not_use_dma));
    config_vector.emplace_back("not_use_mpi_colls",
                               std::to_string(not_use_mpi_colls));
    config_vector.emplace_back("num_levels", std::to_string(num_levels));
    config_vector.emplace_back("iterations", std::to_string(iterations));
    config_vector.emplace_back("hiearchy_aware",
                               std::to_string(hierarchy_aware));
    return config_vector;
  }
};

Parameters read_cli(int argc, char const** argv) {
  Parameters params;
  CLI::App app{"KaDiS Benchmark"};
  app.add_option("--datasize", params.datasize);
  app.add_option("--iterations", params.iterations);
  app.add_option("--num-levels", params.num_levels);
  app.add_option("--imbalance", params.imbalance);
  app.add_flag("--not-use-dma", params.not_use_dma);
  app.add_flag("--not-use-mpi-colls", params.not_use_mpi_colls);
  app.add_flag("--hierarchy-aware", params.hierarchy_aware);
  app.add_option("--json_output_path", params.output_path);
  params.num_pe = kamping::world_size();
  CLI11_PARSE_MPI(app, argc, argv);
  return params;
}

auto gen_data(std::size_t n) {
  using T = std::uint32_t;
  std::vector<T> vec(n);
  std::mt19937 gen(kamping::world_rank());
  std::uniform_int_distribution<T> distrib(0u, 1'000'000u);
  for (std::size_t i = 0; i < n; ++i) {
    vec[i] = distrib(gen);
  }
  return vec;
}

template <typename T>
std::uint64_t sum(std::vector<T> const& data) {
  using namespace kamping;
  std::uint64_t local_sum =
      std::accumulate(data.begin(), data.end(), std::uint64_t{0u});
  return comm_world().allreduce_single(send_buf(local_sum), op(std::plus<>{}));
}

template <typename T>
bool is_sorted(std::vector<T> const& data) {
  using namespace kamping;
  const bool locally_sorted = std::is_sorted(data.begin(), data.end());
  const auto first_last =
      comm_world().allgather(send_buf({data.front(), data.back()}));
  const bool globally_sorted =
      std::is_sorted(first_last.begin(), first_last.end());

  return comm_world().allreduce_single(
      send_buf(locally_sorted && globally_sorted), op(ops::logical_and<>{}));
}

template <typename T>
void run_sorter(std::vector<T>& data, const Parameters& params) {
  std::mt19937_64 gen(kamping::world_rank() + 1092983);
  MPI_Datatype my_mpi_type = kamping::mpi_datatype<T>();
  if (!params.not_use_mpi_colls) {
    RBC::Comm rcomm;
    RBC::Create_Comm_from_MPI(MPI_COMM_WORLD, &rcomm, true, true, true);
    if (kamping::world_rank() == 0) {
      std::cout << "use MPI-Colls: " << rcomm.useMPICollectives() << std::endl;
    }
    if (params.hierarchy_aware) {
      std::vector<std::size_t> ks{kamping::world_size() / 48u, 48u};
      Ams::sort(my_mpi_type, data, ks, gen, rcomm, std::less<T>{},
                params.imbalance, !params.not_use_dma);

    } else {
      Ams::sortLevel(my_mpi_type, data, params.num_levels, gen, rcomm,
                     std::less<T>{}, params.imbalance, !params.not_use_dma);
    }
  } else {
    if (params.hierarchy_aware) {
      std::vector<std::size_t> ks{kamping::world_size() / 48u, 48u};
      Ams::sort(my_mpi_type, data, ks, gen, MPI_COMM_WORLD, std::less<T>{},
                params.imbalance, !params.not_use_dma);

    } else {
      Ams::sortLevel(my_mpi_type, data, params.num_levels, gen, MPI_COMM_WORLD,
                     std::less<T>{}, params.imbalance, !params.not_use_dma);
    }
  }
}

inline void print_as_jsonlist_to_file(std::vector<std::string> objects,
                                      std::string filename) {
  std::ofstream outstream(filename);
  outstream << "[" << std::endl;
  for (std::size_t i = 0; i < objects.size(); ++i) {
    if (i > 0) {
      outstream << "," << std::endl;
    }
    outstream << objects[i];
  }
  outstream << std::endl << "]" << std::endl;
}

int main(int argc, char const** argv) {
  using namespace kamping;
  kamping::Environment e;
  kamping::Communicator comm;

  const auto params = read_cli(argc, argv);

  std::vector<std::string> timer_output;
  std::vector<std::string> counter_output;

  for (std::size_t it = 0; it < params.iterations; ++it) {
    measurements::timer().clear();
    measurements::counter().clear();
    measurements::timer().synchronize_and_start("gen_data");
    auto data = gen_data(params.datasize);
    measurements::timer().stop();
    const auto prev_sum = sum(data);
    comm.barrier();
    measurements::timer().synchronize_and_start("total_time");
    run_sorter(data, params);
    measurements::timer().stop();

    const auto cur_sum = sum(data);
    const bool sorted = is_sorted(data);
    if (comm.is_root()) {
      std::cout << "is sorted: " << sorted
                << " and same sum: " << (cur_sum == prev_sum) << std::endl;
    }

    std::stringstream sstream_counter;
    std::stringstream sstream_timer;
    auto config_vector = params.config();
    config_vector.emplace_back("iteration", std::to_string(it));
    kamping::measurements::SimpleJsonPrinter<double> printer_timer(
        sstream_timer, config_vector);
    kamping::measurements::SimpleJsonPrinter<std::int64_t> printer_counter(
        sstream_counter, config_vector);
    kamping::measurements::timer().aggregate_and_print(printer_timer);
    kamping::measurements::counter().aggregate_and_print(printer_counter);
    timer_output.emplace_back(sstream_timer.str());
    counter_output.emplace_back(sstream_counter.str());
  }
  if (comm.rank() == 0) {
    print_as_jsonlist_to_file(timer_output, params.output_path + "_timer.json");
    print_as_jsonlist_to_file(counter_output,
                              params.output_path + "_counter.json");
  }
}
