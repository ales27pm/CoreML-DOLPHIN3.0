#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {
int add(int a, int b) { return a + b; }
int mul(int a, int b) { return a * b; }
}

PYBIND11_MODULE(libmath, m) {
    m.doc() = "Deterministic math bindings for integration tests";

    m.def("add", &add, py::arg("a"), py::arg("b"),
          R"pbdoc(Returns the sum of two integers.)pbdoc");
    m.def("mul", &mul, py::arg("a"), py::arg("b"),
          R"pbdoc(Returns the product of two integers.)pbdoc");
}
