//
// Created by Emin Tunc Kirimlioglu on 5/30/25.
//

#ifndef MODEL_STUB_HPP
#define MODEL_STUB_HPP

#include <torch/torch.h>

// // Non-templated base for polymorphism
// class ModelStub {
// public:
//    virtual torch::Tensor forward(torch::Tensor x) = 0;
//    virtual ~ModelStub() = default;
// };

// // Templated class that provides cloning + inherits from base
// template<typename Derived>
// class ModelStub : public torch::nn::Cloneable<Derived> {
// public:
//    virtual torch::Tensor forward(torch::Tensor x) = 0;
// };

// template<typename Derived>
// class ModelStub : public torch::nn::Cloneable<Derived>{
// public:
//    // Emin's method, couldn't think of better idea to have common interface
//    // for our other models so here exists a forwardable model implementation
//    virtual ::torch::Tensor forward(::torch::Tensor x) = 0;
// };
#endif //MODEL_STUB_HPP
