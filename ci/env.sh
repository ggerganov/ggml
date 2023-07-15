#!/bin/bash

## helper functions

# download a file if it does not exist or if it is outdated
function gg_wget {
    local out=$1
    local url=$2

    local cwd=`pwd`

    mkdir -p $out
    cd $out

    # should not re-download if file is the same
    wget -N $url

    cd $cwd
}

# useful for exporting bash variables and being able to vertically align them
function gg_export {
    local var=$1
    local val=$2

    if [ -z "${!var}" ]; then
        export $var=$val
    fi
}

## general env

gg_export GG_GGML_DUMMY "dummy"

env | grep GG_GGML | sort

printf "\n"
