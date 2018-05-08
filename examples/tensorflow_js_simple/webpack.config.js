
const path = require('path');

module.exports = {
  mode: 'development',
  entry: './src/index.js',
  output: {
    filename: 'bundle.js',
    path: path.join(__dirname, 'public')
  },
  module: {
    rules: [{
      test: /\.js$/,
      exclude: /node_modules/,
    }],
  },
  devServer: {
    contentBase: path.join(__dirname, 'public'),
    // Uncomment the following two lines to enable access from a remote device.
    // Note that a Service Worker doesn't work when served remotely via http,
    // as secure origins are required for its registration.
    // host: '0.0.0.0',
    // disableHostCheck: true
  }
};
