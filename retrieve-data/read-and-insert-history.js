const csv = require('csvtojson');

let csvFilePath = '../data/companies-history-0-2017-02-01.csv';

csv()
  .fromFile(csvFilePath)
  .on('json', (jsonObj) => {
    console.log(JSON.stringify(jsonObj));
  })
  .on('done', (error) => {
    if (error) {
      throw error;
    }
    console.log('end');
  });
