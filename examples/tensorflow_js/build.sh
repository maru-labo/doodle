#!/usr/bin/env bash

pushd dist
tar cvzf ../tensorflowjs.doodle.marulabo.tar.gz index.html buefy.min.*.css marulabo.*.svg src.*.js
popd

