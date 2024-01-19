#ifndef SIMULATOR
#define SIMULATOR
#include <boost/program_options.hpp>
#include <iostream>
#include "NeuronNetwork.hpp"
#include "signal.h"
#include "boost/asio.hpp"
struct Probes
{
    public:
        std::map<std::string, std::string> electrodes;
};

class Simulator
{
    public:
        Simulator(int argc, char* argv[]) 
        {
            boost::program_options::variables_map vm;
            boost::program_options::options_description desc("Simulation Setup");
            desc.add_options() 
                ("simulation,s", boost::program_options::value<std::string>(), "simulation config file name")
                ("neurons,n", boost::program_options::value< std::vector<std::string> >() ->multitoken() , "neuron config files name")
                ("probes,p", boost::program_options::value<std::string>(), "probes config file name")
                ("help,h", "help screen")
            ;
            boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm); 
            boost::program_options::notify(vm);
            if (vm.count("help")) 
            {
                std::cout << desc << "\n";
                exit(EXIT_SUCCESS);
            }
            std::vector< std::string > neuron_cfg;
            std::string probe_cfg;
            if (vm.count("simulation"))
            {
                std::string sim_cfg = vm["simulation"].as<std::string>();
                std::ifstream file(sim_cfg + ".json");
                nlohmann::json data = nlohmann::json::parse(file);
                dt = data["time"].get<double>();
                sim_time_ratio = data["simulation time ratio"].get<double>();
                sim_end_time = data["simulation end time"].get<int>();
            }
            if (vm.count("neurons")) neuron_cfg = vm["neurons"].as<std::vector<std::string>>();
            if (vm.count("probes")) 
            {
                probe_cfg = vm["probes"].as<std::string>();
                std::ifstream file(probe_cfg + ".json");
                nlohmann::json data = nlohmann::json::parse(file);
                probes_file.open(data["file"].get<std::string> ());
                signal(SIGINT, Simulator::signal_callback_handler);
                probes.electrodes = data["probes"].get< std::map< std::string, std::string > >();
                probes_file << "time[s]";
                for (auto iter = probes.electrodes.begin(); iter != probes.electrodes.end(); iter++)
                    probes_file << "," << iter->first << "-" << iter->second;
                probes_file << "\n";
            }
            NeuronNetwork network = NeuronNetwork(neuron_cfg);
            boost::asio::io_context timer_io(1);
            unsigned int sleep_us = dt * 1000000 / sim_time_ratio;
            for (int i = 1; i * dt < sim_end_time ; i++) 
            {
                boost::asio::deadline_timer timer(timer_io, boost::posix_time::microseconds(sleep_us));
                network.spinOnce(dt);
                probes_file << i * dt;
                for (auto iter = probes.electrodes.begin(); iter != probes.electrodes.end(); iter++)
                    probes_file << "," << network.probe(iter->first) - network.probe(iter->second);
                probes_file << "\n";
                timer.wait();
            }
            probes_file.close();
            exit(EXIT_SUCCESS);
        }

    private:
        double dt = 0.001;
        double sim_time_ratio = 1.0;
        int sim_end_time = -1;
        static std::ofstream probes_file;
        Probes probes;
        NeuronNetwork network;
        static void signal_callback_handler(int signum) 
        {
            std::cout << "file saved\n" ;
            probes_file.close();
            exit(EXIT_SUCCESS);
        }
};

std::ofstream Simulator::probes_file;

#endif