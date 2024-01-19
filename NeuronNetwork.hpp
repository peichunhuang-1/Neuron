#ifndef NEURON_NETWORK
#define NEURON_NETWORK
#include "Neuron.hpp"
struct NeuronInfo
{
    public:
        NeuronInfo() {}
        NeuronInfo(std::string _name, std::shared_ptr<EmptyNeuron> _neuron, unsigned int _N, NeuronType _type) 
        : name(_name), neuron(_neuron), N(_N), type(_type)
        {}
        std::string name;
        std::shared_ptr<EmptyNeuron> neuron;
        unsigned int N;
        NeuronType type;
        std::vector<std::string> connections;
};

template<class T>
NeuronInfo create_neuron(std::string name, T n)
{
    return NeuronInfo(name, std::make_shared<T>(n), n.N(), n.type());
};

class NeuronNetwork
{
    public:
        NeuronNetwork() {}
        NeuronNetwork(std::vector<std::string> cfgs) 
        {
            for (auto cfg : cfgs)
            {
                std::ifstream file(cfg + ".json");
                nlohmann::json data = nlohmann::json::parse(file);
                for (auto& el : data.items())
                {
                    unsigned int N = el.value()["N"].get<unsigned int> ();
                    NeuronInfo info;
                    switch (el.value()["type"].get<unsigned int> () )
                    {
                        case CONST:
                        {
                            ConstNeuron const_neuron(N);
                            if (el.value()["weights"].get< std::vector<double> >().data() != nullptr) 
                                const_neuron.weights( el.value()["weights"].get< std::vector<double> >().data() );
                            const_neuron.c = el.value()["c"].get<double>();
                            info = create_neuron(el.key(), const_neuron);
                            info.connections = el.value()["connection"].get< std::vector<std::string> >();
                            break;
                        }
                        case MATSUOKA:
                        {
                            MatsuokaNeuron matsuoka_neuron(N);
                            if (el.value()["weights"].get< std::vector<double> >().data() != nullptr) 
                                matsuoka_neuron.weights( el.value()["weights"].get< std::vector<double> >().data() );
                            matsuoka_neuron.s = el.value()["s"].get<double>();
                            matsuoka_neuron.b = el.value()["b"].get<double>();
                            matsuoka_neuron.tu = el.value()["tu"].get<double>();
                            matsuoka_neuron.tv = el.value()["tv"].get<double>();
                            info = create_neuron(el.key(), matsuoka_neuron);
                            info.connections = el.value()["connection"].get< std::vector<std::string> >();
                            break;
                        }
                        case SIGMOID:
                        {
                            SigmoidNeuron sigmoid_neuron(N);
                            if (el.value()["weights"].get< std::vector<double> >().data() != nullptr) 
                                sigmoid_neuron.weights( el.value()["weights"].get< std::vector<double> >().data() );
                            sigmoid_neuron.c = el.value()["c"].get<double>();
                            sigmoid_neuron.a = el.value()["a"].get<double>();
                            info = create_neuron(el.key(), sigmoid_neuron);
                            info.connections = el.value()["connection"].get< std::vector<std::string> >();
                            break;
                        }
                        case TIMER:
                        {
                            TimerNeuron timer_neuron(N);
                            if (el.value()["weights"].get< std::vector<double> >().data() != nullptr) 
                                timer_neuron.weights( el.value()["weights"].get< std::vector<double> >().data() );
                            timer_neuron.c = el.value()["c"].get<double>();
                            timer_neuron.end = el.value()["end"].get<double>();
                            timer_neuron.start = el.value()["start"].get<double>();
                            info = create_neuron(el.key(), timer_neuron);
                            info.connections = el.value()["connection"].get< std::vector<std::string> >();
                            break;
                        }
                        case INTER:
                        {
                            InterNeuron inter_neuron(N);
                            if (el.value()["weights"].get< std::vector<double> >().data() != nullptr) 
                                inter_neuron.weights( el.value()["weights"].get< std::vector<double> >().data() );
                            inter_neuron.bypass = el.value()["bypass"].get<unsigned int>();
                            info = create_neuron(el.key(), inter_neuron);
                            info.connections = el.value()["connection"].get< std::vector<std::string> >();
                            break;
                        }
                        case THRES:
                        {
                            ThresNeuron thres_neuron(N);
                            if (el.value()["weights"].get< std::vector<double> >().data() != nullptr) 
                                thres_neuron.weights( el.value()["weights"].get< std::vector<double> >().data() );
                            thres_neuron.thres = el.value()["thres"].get<double>();
                            info = create_neuron(el.key(), thres_neuron);
                            info.connections = el.value()["connection"].get< std::vector<std::string> >();
                            break;
                        }
                    default:
                        break;
                    }
                    Network[el.key()] = info;
                }
            }

        }
        
        void spinOnce(double dt) 
        {
            for (auto iter = Network.begin(); iter != Network.end(); iter++) 
            {
                probes[iter->first] = iter->second.neuron->y(dt);
            }
            for (auto iter = Network.begin(); iter != Network.end(); iter++)
            {
                std::vector<double> synapses;
                for (auto synapse_name: iter->second.connections)
                {
                    synapses.push_back(probes[synapse_name]);
                }
                iter->second.neuron->synapses(synapses.data());
            }
        }
        double probe(std::string name) {return probes[name];}
    private:
        std::map<std::string, NeuronInfo> Network; 
        std::map<std::string, double> probes = {{"GND", 0}, }; 
        
};


#endif