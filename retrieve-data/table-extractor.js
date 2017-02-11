'use strict';

const fs = require('fs');
const jsdom = require('jsdom');
const symbols = require('./symbols.json');
const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');

let companyHistory = {};

let convertTable = function(tableDocObj, symbolVal) {
  let tableAsJson = {};
  let yearMonth = [];

  let rows = tableDocObj.document.querySelectorAll('tr');

  // console.log(rows);

  if (rows) {
    for (let c = 0; c < rows.length; c++) {
      // console.log('----------');
      let resultVals = rows[c].innerHTML
        .replace(/<\/td>/g, '||')
        .replace(/<td.*>/g, '')
        .replace(/\n/g, '');
      // console.log(resultVals);
      let resultArray = resultVals.split('||');
      /* Remove the last value as there is a closing <td> which will result
      in an empty value */

      resultArray = resultArray.slice(0, resultArray.length - 1);

      for (let r = 0; r < resultArray.length; r++) {
        resultArray[r] = (resultArray[r]).trim();
      }
      // console.log(JSON.stringify(resultArray));

      if (c === 0) {
        /* Header row
         * Set year/month lookup values */
        yearMonth = resultArray;

        /* Retrieve year/month key values - skip first element as it will be
        blank */
        for (let r = 1; r < resultArray.length; r++) {
          // Initialise year / month key value if not present
          if (!tableAsJson[resultArray[r]]) {
            // tableAsJson[resultArray[r]] = [];
            tableAsJson[resultArray[r]] = {};
          }
        }
      } else {
        /* Values row
        * First element is value name, subsequent entries are values for
        year/month */
        let valueName = resultArray[0];

        /* Other elements contain numeric values, add each one to the
        appropriate year, month */
        for (let r = 1; r < resultArray.length; r++) {
          let valObj = {};
          valObj[valueName] = resultArray[r];
          // tableAsJson[yearMonth[r]].push(valObj);
          tableAsJson[yearMonth[r]][valueName] = resultArray[r];
        }
      }
    }
    // console.log(JSON.stringify(tableAsJson));

    // Initialise company key value if not present
    if (!companyHistory[symbolVal]) {
      companyHistory[symbolVal] = {};
    }

    // Loop through year/month keys
    Object.keys(tableAsJson).forEach((yearKey) => {
      // Check if yearmonh val exists, if not initilalise it
      if (!companyHistory[symbolVal][yearKey]) {
        // companyHistory[symbolVal][yearKey] = [];
        companyHistory[symbolVal][yearKey] = {};
      }

      // Add results from this table to the main tableAsJson
      /* if (companyHistory[symbolVal][yearKey].length === 0) {
        companyHistory[symbolVal][yearKey] = tableAsJson[yearKey];
      } else { */
      /* companyHistory[symbolVal][yearKey] = companyHistory[symbolVal][yearKey]
        .concat(tableAsJson[yearKey]);*/

      Object.keys(tableAsJson[yearKey]).forEach((yearVal) => {
        companyHistory[symbolVal][yearKey][yearVal] = tableAsJson[yearKey][yearVal];
      });
    // }
    });
  }
};


let domHTML = function(html) {
  return new Promise(function(resolve, reject) {
    jsdom.env(html, function(err, window) {
      if (err) {
        reject(err);
      }

      resolve(window);
    });
  });
};

let parseHTML = function(html) {
  return new Promise(function(resolve, reject) {
    domHTML(html).then((docWindow) => {
      let dataTables = docWindow.document
        .querySelectorAll('.table1.dividendhisttable');

      // free memory associated with the window
      docWindow.close();

      resolve(dataTables);
    }).catch((err) => {
      reject(err);
    });
  });
};

let extractHTML = function(filePath) {
  return new Promise(function(resolve, reject) {
    resolve(fs.readFileSync(filePath, 'utf8'));
  });
};

let writeJSON = function(jsonContent) {
  fs.writeFile('../data/company-history.json',
    JSON.stringify(jsonContent), function(err) {
      if (err)
        throw err;
      console.log('File saved');
    });
};


let runExtraction = asyncify(function() {
  // Generate list of comapny html files
  let symbolVals = [];
  symbols.companies.forEach((company) => {
    // Skip companies up to AAEDC
    if (fs.existsSync('../data/html/' + company.symbol + '.htm')) {
      symbolVals.push(company.symbol);
    }
  });


  symbolVals.forEach((symbolVal) => {
    awaitify(extractHTML('../data/html/' + symbolVal + '.htm')
      .then((html) => {
        return parseHTML(html);
      })
      .then((dataTables) => {
        dataTables.forEach((table) => {
          // console.log(table.outerHTML);
          domHTML('<html>' + table.outerHTML + '</html>')
            .then((extractResult) => {
              convertTable(extractResult, symbolVal);
              writeJSON(companyHistory);
            });
        });
      })
      .catch((err) => {
        console.log(err);
      })
    );
  });

/* Promise.all(extractPromises)
  .then((extractResults) => {
    extractResults.forEach((extractResult) => {
      convertTable(extractResult, symbolVal);
      console.log(JSON.stringify(companyHistory));
    });

  // console.log(JSON.stringify(companyHistory));
  })
  .catch((err) => {
    console.log(err);
  });*/
});

runExtraction();
