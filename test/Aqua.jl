using Aqua
using SOSAT

Aqua.test_all(
  SOSAT;
  ambiguities=(broken=true),
  stale_deps=(),
  deps_compat=(),
  piracies=false,
)
