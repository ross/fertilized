# This script contains Python's venv management features common to all shell
# scripts located in this directory and to the repository pre-commit hook.
# This script is *not* meant to be run directly.

# Exit on any error
set -e

ROOT=$(dirname $(dirname $0))

# Change to path base directory
cd "${ROOT}"

# If no venv name is set, set it to "env"
if [ -z "${VENV_NAME}" ]; then
  VENV_NAME="${ROOT}/env"
fi

ACTIVATE="${VENV_NAME}/bin/activate"
# Check that [venv_directory]/bin/activate exists.
if [ ! -f "${ACTIVATE}" ]; then
  echo "${ACTIVATE} does not exist. Run ./script/bootstrap" >&2
  exit 1
fi

# Activate venv.
source "${ACTIVATE}"
