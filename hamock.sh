#!/bin/bash

if [ ! -z "${HAMMOCK_DOWNLOAD_URL}" ]
then
	curl -o state_replay.log  "${HAMMOCK_DOWNLOAD_URL}"
fi

exec ./hamock.py $*
