
#pragma once

#include "extensions/nvdsinterface/interfaces.hpp"

namespace hm {
namespace extensions {

class HockeyMomComponent : INvDsComponent {
 public:
  // Public methods using which other components can interact with this
  // component via its handle.
  void simpleComponentMethod() {
    //
  }

 private:
  gxf_result_t registerInterface(nvidia::gxf::Registrar *registrar) override {
    nvidia::gxf::Expected<void> result;

    result &= registrar->parameter(
        simple_param_,       // Parameter member variable
        "simple-param-key",  // Parameter name(key). This is used to set
                             // parameter value in a graph
        "Simple Parameter",  // Parameter head line
        "Description of the simple parameter",  // A description of the
                                                // parameter
        100UL,                        // A default value for the parameter
        GXF_PARAMETER_FLAGS_OPTIONAL  // Parameter flags marking it
    );

    result &= registrar->parameter(handle_param_, "handle-parameter",
                                   "Handle Parameter",
                                   "Description of the handle parameter",
                                   std::nullopt, GXF_PARAMETER_FLAGS_OPTIONAL);

    return nvidia::gxf::ToResultCode(result);
  }

  gxf_result_t initialize() override {
    // This method can be used to initialize the component.
    //...

    // Check if parameter is set
    if (simple_param_.try_get() != std::nullopt) {
      uint64_t simple_param_value = simple_param_.try_get().value();
      //...
    }

    return GXF_SUCCESS;  // return GXF_FAILURE in case of any fatal error
  }

  gxf_result_t deinitialize() override {
    // This method can be used to deinitialize the component.
    //...

    return GXF_SUCCESS;  // return GXF_FAILURE in case of any fatal error
  }

  gxf_result_t start() override {
    // Start the component. The underlying DeepStream pipeline and other
    // components are already initialized. It is safe to call methods of other components
    // via their handles.
    //...

    // Check if any component is attached to the parameter
    if (handle_param.try_get() != std::nullopt) {
      SampleOtherComponent *other_comp = handle_param.try_get().value();
      other_comp->otherComponentMethod();
      //...
    }

    return GXF_SUCCESS;  // return GXF_FAILURE in case of any fatal error
  }

  gxf_result_t stop() override {
    // Pipeline has been stopped. All the components would be deinitialized
    // after this. Add any logic for stopping the component.

    return GXF_SUCCESS;  // return GXF_FAILURE in case of any fatal error
  }

  nvidia::gxf::Parameter<uint64_t> simple_param_;
  nvidia::gxf::Parameter<nvidia::gxf::Handle<SampleOtherComponent>>
      handle_param_;
};

}  // namespace test
}  // namespace sample
  