
#ifndef CINM_PLATFORM_ATTRS
#define CINM_PLATFORM_ATTRS

include "CinmBase.td"
include "CinmPlatformAttrInterface.td"

def CinmHostPlatformAttr: Cinm_Attr<"HostPlatform", [
    CinmPlatformAttrInterface
  ]> {
  let summary = "Host platform description";
  let hasCustomAssemblyFormat = 1;
  let mnemonic = "host_platform";
  let description = [{
    Host platform.
  }];

  let extraClassDeclaration = [{
    llvm::StringRef getName() const {
      return "host";
    }
    llvm::ArrayRef<cinm::CinmLevelDefAttr> getLevels() const {
      return {};
    }
  }];

}


def CinmAvailablePlatformsAttr : ArrayOfAttr<Cinm_Dialect, "CinmPlatformArray", "platforms",
                            CinmPlatformAttrInterface.storageType> {}

#endif
