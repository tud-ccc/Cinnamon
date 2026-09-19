/// Fork server mode of cinm-opt.
///
/// Starting cinm-opt costs most of a second: loading several hundred shared
/// libraries, running their static constructors, and registering every
/// dialect and pass. `cinm-opt --serve <socket>` pays that once, then forks a
/// worker per request received from cinm-opt-client. Each worker is an
/// ordinary cinm-opt process that has already been initialized.
///
/// The server re-executes itself when any file it has mapped (the executable
/// or a shared library) changes on disk, so requests after a rebuild run the
/// rebuilt code. It exits after a period without requests.
///
/// @file
/// @author      Clément Fournier (clement.fournier@tu-dresden.de)

#ifndef CINM_OPT_OPTSERVER_H
#define CINM_OPT_OPTSERVER_H

#include <llvm/ADT/STLFunctionalExtras.h>

namespace cinm_opt {

/// Runs one request in a forked worker process and returns its exit code.
using RunRequestFn = llvm::function_ref<int(int argc, char **argv)>;

/// Serves requests on the socket named by `argv[2]` (`argv[1]` is "--serve")
/// until idle. `argv` is also what the server re-executes itself with.
/// Returns the process exit code.
int runServer(int argc, char **argv, RunRequestFn runRequest);

} // namespace cinm_opt

#endif // CINM_OPT_OPTSERVER_H
