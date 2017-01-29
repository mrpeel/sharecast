module.exports = {
  writeToCsv: function(csvData, csvFields, filePrefix) {
    let csvOutput = json2csv({
      data: csvData,
      fields: csvFields,
    });

    let today = new Date();

    fs.writeFile('./data/' + filePrefix + '-' + today.getFullYear() + '-' +
      ('0' + (today.getMonth() + 1)).slice(-2) + '-' +
      ('0' + today.getDate()).slice(-2) + '.csv', csvOutput, function(err) {
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
