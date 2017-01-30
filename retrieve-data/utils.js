const json2csv = require('json2csv');
const fs = require('fs');
const symbols = require('./symbols.json');
const lastRetrieval = require('./last-retrieval.json');


module.exports = {
  getLastRetrievalDate: function() {
    return lastRetrieval['retrieval-date'];
  },
  getCompanies: function() {
    return symbols.companies;
  },
  getIndices: function() {
    return symbols.indices;
  },
  returnDateAsString: function(dateValue) {
    let checkDate = new Date(dateValue);

    return checkDate.getFullYear() + '-' +
      ('0' + (checkDate.getMonth() + 1)).slice(-2) + '-' +
      ('0' + checkDate.getDate()).slice(-2);
  },
  createFieldArray: function(fieldObject) {
    return Object.keys(fieldObject);
  },
  writeToCsv: function(csvData, csvFields, filePrefix, dateString) {
    // If no date string supplied, use today's date
    dateString = dateString || today.getFullYear() + '-' +
      ('0' + (today.getMonth() + 1)).slice(-2) + '-' +
      ('0' + today.getDate()).slice(-2);

    let csvOutput = json2csv({
      data: csvData,
      fields: csvFields,
    });

    let today = new Date();

    fs.writeFile('../data/' + filePrefix + '-' + dateString + '.csv',
      csvOutput, function(err) {
        if (err)
          throw err;
        console.log('File saved');
      });
  },
  checkForNumber: function(value) {
    let finalChar = String(value).slice(-1);
    let leadingVal = String(value).slice(0, String(value).length - 1);
    let charsToMatch = {
      'k': 1000,
      'K': 1000,
      'm': 1000000,
      'M': 1000000,
      'b': 1000000000,
      'B': 1000000000,
    };
    let possibleChars = Object.keys(charsToMatch);

    // Check if final character is thousands, millions or billions
    if (!isNaN(leadingVal) && possibleChars.indexOf(finalChar) > -1) {
      // if it is, multiple value to get normal number
      return leadingVal * charsToMatch[finalChar];
    } else {
      return value;
    }
  },
};
