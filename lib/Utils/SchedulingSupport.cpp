#include <cinm-mlir/Utils/Scheduling/SchedulingSupport.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>

using namespace mlir;
using namespace mlir::cinm::utils;

void cinm::utils::evolveName(llvm::StringRef name, std::string &result) {
  size_t last_char_pos = name.find_last_not_of("0123456789");
  StringRef baseName = name.substr(0, last_char_pos + 1);
  StringRef number = name.substr(last_char_pos + 1, name.size());
  if (number.empty()) {
    result.append(baseName);
    result.append("1");
  } else {
    int num = 0;
    number.getAsInteger(10, num);
    num++;
    result.append(baseName);
    result.append(std::to_string(num));
  }
}

NameInventor NameInventor::getNameInventor(Operation *loc,
                                           llvm::StringRef hint) {
  NameInventor::SetType existingNames;

  Block *block = loc->getBlock();
  if (!block && loc->getParentRegion())
    block = &loc->getParentRegion()->front();

  if (block) {
    for (auto &op : *block) {
      op.walk([&](Operation *inner) {
        for (auto namedAttr : inner->getAttrDictionary()) {
          if (auto strAttr = llvm::dyn_cast<StringAttr>(namedAttr.getValue())) {
            if (strAttr.getValue().starts_with(hint))
              existingNames.insert(strAttr.getValue());
          }
        }
      });
    }
  }

  return NameInventor(std::move(existingNames), loc->getContext(), hint);
}

StringAttr NameInventor::getUniqueName(StringRef hint) {
  unsigned num = 0;
  std::string str;
  str.append(prefix);
  str.append(hint);

  const size_t sizeBeforeNum = str.size();

  while (usedNames.contains(str)) {
    str.erase(sizeBeforeNum, str.size());
    num++;
    str += std::to_string(num);
  }

  auto result = StringAttr::get(context, str);
  usedNames.insert(result);
  return result;
}
