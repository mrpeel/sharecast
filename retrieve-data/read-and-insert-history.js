const csv = require('csvtojson');
const utils = require('./utils');

let csvFilePath = '../data/companies-history-0-2017-02-01.csv';

let retrieveCsv = function(filePath) {
  csv()
    .fromFile(filePath)
    .on('json', (jsonObj) => {
      return jsonObj;
    })
    .on('done', (error) => {
      if (error) {
        throw error;
      }
      console.log('end');
    });
};


utils.getLastRetrievalDate().then((dateVal) => {
  console.log(dateVal);
});
