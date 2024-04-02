// This program demonstrates how to use collie::tf::DataPipeline to create
// a pipeline with in-pipe data automatically managed by the Taskflow
// library.

#include <collie/taskflow/taskflow.h>
#include <collie/taskflow/algorithm/data_pipeline.h>

int main() {

  // dataflow => void -> int -> std::string -> float -> void 
  collie::tf::Taskflow taskflow("pipeline");
  collie::tf::Executor executor;

  const size_t num_lines = 3;
  
  // create a pipeline graph
  collie::tf::DataPipeline pl(num_lines,
    collie::tf::make_data_pipe<void, int>(collie::tf::PipeType::SERIAL, [&](collie::tf::Pipeflow& pf) {
      if(pf.token() == 5) {
        pf.stop();
        return 0;
      }
      else {
        printf("first pipe returns %zu\n", pf.token());
        return static_cast<int>(pf.token());
      }
    }),

    collie::tf::make_data_pipe<int, std::string>(collie::tf::PipeType::SERIAL, [](int& input) {
      printf("second pipe returns a strong of %d\n", input + 100);
      return std::to_string(input + 100);
    }),

    collie::tf::make_data_pipe<std::string, void>(collie::tf::PipeType::SERIAL, [](std::string& input) {
      printf("third pipe receives the input string %s\n", input.c_str());
    })
  );

  // build the pipeline graph using composition
  taskflow.composed_of(pl).name("pipeline");

  // dump the pipeline graph structure (with composition)
  taskflow.dump(std::cout);

  // run the pipeline
  executor.run(taskflow).wait();

  return 0;
}

