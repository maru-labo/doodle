
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
      use: [{
        loader: 'babel-loader',
        options: {
          presets: ['env']
        }
      }],
    }],
  },
  devServer: {
    contentBase: path.join(__dirname, 'public'),
    // enable access from a remote device
    host: '0.0.0.0',
    disableHostCheck: true
  }
};
