
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
    // Uncomment the last three lines to enable access from a remote device.
    // Note that https is enabled as a Service Workers require secure origins.
    // The server uses a self-signed certificate by default, which will result
    // in displaying security warnings in browser.
    // See: https://webpack.js.org/configuration/dev-server/#devserver-https
    // host: '0.0.0.0',
    // disableHostCheck: true,
    // https: true
  }
};
