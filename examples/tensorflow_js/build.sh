#!/usr/bin/env bash

pushd dist
tar cvzf ../tensorflowjs.doodle.marulabo.tar.gz index.html buefy.min.css index.js
popd

