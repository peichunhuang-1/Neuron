#ifndef NEURON
#define NEURON
#include "stdlib.h"
#include "stdio.h"
#include <algorithm>
#include <vector>
#include <iostream>
#include <map>
#include "nlohmann/json.hpp"
#include "fstream"

#define CREATE_NEURON(var)  create_neuron(#var, var)

enum NeuronType
{
    EMPTY = 0,
    CONST = 1,
    MATSUOKA = 2,
    SIGMOID = 3,
    TIMER = 4,
    INTER = 5,
    THRES = 6,
};

class EmptyNeuron
{
    public:
        EmptyNeuron() = default;
        ~EmptyNeuron() = default;
        virtual NeuronType type() {return EMPTY;}
        virtual unsigned int N() {return 0;}
        virtual void synapses(double s[]) {}
        virtual void weights(double w[]) {}
        virtual double y(double dt) {return 0;}
};

class ConstNeuron : public EmptyNeuron
{
    public:
        ConstNeuron(unsigned int _n) : n(_n) 
        {
            if (n > 0)
            {
                synapse = new double [n] {0, };
                weight = new double [n] {0, };
            }
            else 
            {
                synapse = nullptr;
                weight = nullptr;
            }
        }
        virtual ~ConstNeuron() = default;
        virtual double y( double dt) override
        {
            double sum = 0;
            for (unsigned int i = 0; i < n; i++)
                sum += weight[i] * synapse[i];
            return sum + c;
        }
        virtual NeuronType type() override {return CONST;}
        virtual void synapses(double _s[]) override{memcpy(synapse, _s, n*(sizeof(double)));}
        virtual void weights(double _w[]) override {memcpy(weight, _w, n*(sizeof(double)));}
        double c = 0;
        virtual unsigned int N() override {return n;}

    private:
        unsigned int n = 0;
        double *weight;
        double *synapse;

};

class MatsuokaNeuron : public EmptyNeuron
{
    public:
        MatsuokaNeuron(unsigned int _n) : n(_n)
        {
            u = - (double) (rand() / (RAND_MAX + 1.0));
            v = - (double) (rand() / (RAND_MAX + 1.0));
            if (n > 0)
            {
                synapse = new double [n] {0, };
                weight = new double [n] {0, };
            }
            else 
            {
                synapse = nullptr;
                weight = nullptr;
            }
        }
        virtual ~MatsuokaNeuron() = default;

        virtual double y(double dt) override
        {
            double du = dudt() * dt;
            double dv = dvdt() * dt;
            u += du;
            v += dv;
            return u > 0 ? u : 0; 
        }
        virtual NeuronType type() override {return MATSUOKA;}
        virtual void synapses(double _s[]) override{memcpy(synapse, _s, n*(sizeof(double)));}
        virtual void weights(double _w[]) override {memcpy(weight, _w, n*(sizeof(double)));}
        virtual unsigned int N() override {return n;}

        double s = 1.0;
        double tu = 1.0;
        double tv = 1.0;
        double b = 5;
        
    private:
        unsigned int n;
        double *weight;
        double *synapse;
        double u = 0.0;
        double v = 0.0;
        
        double dudt() 
        {
            double sum = 0;
            for (unsigned int i = 0; i < n; i++)
                sum += weight[i] * synapse[i];
            return (-u + sum + s - b * v) / tu;
        }
        double dvdt() 
        {
            return (u > 0 ? u - v : - v) / tv;
        }
};

class SigmoidNeuron : public EmptyNeuron
{
    public:
        SigmoidNeuron(unsigned int _n) : n(_n) 
        {
            if (n > 0)
            {
                synapse = new double [n] {0, };
                weight = new double [n] {0, };
            }
            else 
            {
                synapse = nullptr;
                weight = nullptr;
            }
        }
        virtual ~SigmoidNeuron() = default;
        virtual double y( double dt) override
        {
            double sum = 0;
            for (unsigned int i = 0; i < n; i++)
                sum += weight[i] * synapse[i];
            return sigmoid(sum) + c;
        }
        virtual NeuronType type() override {return SIGMOID;}
        virtual void synapses(double _s[]) override{memcpy(synapse, _s, n*(sizeof(double)));}
        virtual void weights(double _w[]) override {memcpy(weight, _w, n*(sizeof(double)));}
        double c = 0;
        double a = 1.;
        virtual unsigned int N() override {return n;}

    private:
        unsigned int n = 0;
        double *weight;
        double *synapse;
        double sigmoid(double x)
        {
            return 1. / (1. + exp(- a * x));
        }

};

class TimerNeuron : public EmptyNeuron
{
    public:
        TimerNeuron(unsigned int _n) : n(_n) 
        {
            if (n > 0)
            {
                synapse = new double [n] {0, };
                weight = new double [n] {0, };
            }
            else 
            {
                synapse = nullptr;
                weight = nullptr;
            }
        }
        virtual ~TimerNeuron() = default;
        virtual double y( double dt) override
        {
            double sum = 0;
            for (unsigned int i = 0; i < n; i++)
                sum += weight[i] * synapse[i];
            count += dt;
            return count > start? count < end ? (sum + c): 0 : 0;
        }
        virtual NeuronType type() override {return TIMER;}
        virtual void synapses(double _s[]) override{memcpy(synapse, _s, n*(sizeof(double)));}
        virtual void weights(double _w[]) override {memcpy(weight, _w, n*(sizeof(double)));}
        double c = 0;
        double end = 1;
        double start = 0;
        virtual unsigned int N() override {return n;}

    private:
        double count = 0;
        unsigned int n = 0;
        double *weight;
        double *synapse;
};

class InterNeuron : public EmptyNeuron
{
    public:
        InterNeuron(unsigned int _n) : n(_n) 
        {
            if (n > 0)
            {
                synapse = new double [n] {0, };
                weight = new double [n] {0, };
            }
            else 
            {
                synapse = nullptr;
                weight = nullptr;
            }
        }
        virtual ~InterNeuron() = default;
        virtual double y( double dt) override
        {
            double sum = 0;
            for (unsigned int i = 0; i < n; i++)
                if (i != bypass) sum += weight[i] * synapse[i];
            return sum > 0? weight[bypass] * synapse[bypass] : 0;
        }
        virtual NeuronType type() override {return INTER;}
        virtual void synapses(double _s[]) override{memcpy(synapse, _s, n*(sizeof(double)));}
        virtual void weights(double _w[]) override {memcpy(weight, _w, n*(sizeof(double)));}
        virtual unsigned int N() override {return n;}
        unsigned int bypass = 0;
    private:
        unsigned int n = 0;
        double *weight;
        double *synapse;
};

class ThresNeuron : public EmptyNeuron
{
    public:
        ThresNeuron(unsigned int _n) : n(_n) 
        {
            if (n > 0)
            {
                synapse = new double [n] {0, };
                weight = new double [n] {0, };
            }
            else 
            {
                synapse = nullptr;
                weight = nullptr;
            }
        }
        virtual ~ThresNeuron() = default;
        virtual double y( double dt) override
        {
            double sum = 0;
            for (unsigned int i = 0; i < n; i++)
                sum += weight[i] * synapse[i];
            return sum > thres? sum : 0;
        }
        virtual NeuronType type() override {return THRES;}
        virtual void synapses(double _s[]) override{memcpy(synapse, _s, n*(sizeof(double)));}
        virtual void weights(double _w[]) override {memcpy(weight, _w, n*(sizeof(double)));}
        virtual unsigned int N() override {return n;}
        double thres = 0;
    private:
        unsigned int n = 0;
        double *weight;
        double *synapse;
};

#endif