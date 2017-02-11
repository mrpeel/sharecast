'use strict';
const fetch = require('node-fetch');
const credentials = require('../credentials/get-history.json');
const nightmareElectron = require('nightmare');
const xray = require('x-ray');
const tabletojson = require('tabletojson');
const fs = require('fs');
const symbols = require('./symbols.json');

const nightmare = nightmareElectron();
let x = xray().driver(nightmare);

let get = function get(url) {
  return new Promise(function(resolve, reject) {
    console.log(url);
    fetch(url)
      .then((response) => {
        if (response.ok) {
          return response.text();
        } else {
          reject(response.statusText);
        }
      })
      .then((responseBody) => {
        console.log(responseBody);
        if (responseBody.indexOf('<table>') !== -1) {
          xray(responseBody, ['table@html'])(function(conversionError,
            tableHtmlList) {
            if (conversionError) {
              return reject(conversionError);
            }
            resolve(tableHtmlList.map(function(table) {
              /* xray returns the html inside each table tag, and tabletojson
               * expects a valid html table, so we need to re-wrap the table.
               * Returning the first element in the converted array because
               * we should only ever be parsing one table at a time within this
                 map. */
              return tabletojson.convert('<table>' + table + '</table>')[0];
            }));
          });
        } else {
          reject('No tables found in page');
        }
      });
  });
};


let runNightmare = function() {
  let companySymbols = [];
  symbols.companies.forEach((company) => {
    // SKip companies up to AAEDC
    if (!fs.existsSync('../data/html/' + company.symbol + '.htm')) {
      companySymbols.push(company.symbol);
    }
  });

  console.log(credentials['login-url']);
  nightmare
    .goto(credentials['login-url']);
  console.log(credentials['login-button']);
  nightmare.click(credentials['loginButton']);
  nightmare.wait('input[type=email]');
  console.log(credentials['email']);
  nightmare.type('input[type=email]', credentials['email']);
  console.log(credentials['password']);
  nightmare.type('input[type=password]', credentials['password']);
  nightmare.click('#LoginSubmit');
  nightmare.wait(3000);
  companySymbols.forEach((symbol) => {
    // Wait for a random period between 5 - 10 seconds
    nightmare.wait((5000 + (Math.random() * 5000)));
    nightmare
      .goto(credentials['base-url'] + symbol);

    nightmare.html('../data/html/' + symbol + '.htm', 'HTMLOnly');
  });

  nightmare.end()
    .then(() => {
      console.log('Doned');
    }).catch((err) => {
    console.log(err);
  });

/*    .then((tables) => {
      console.log(tables);
      tables.forEach((table) => {
        tableJson = tableHtmlList.map(function(table) {
          return tabletojson.convert(table.outerHTML)[0];
        });

        console.log(JSON.stringify(tableJson));
      })
        .catch((err) => {
          console.log(err);
        });
    });*/
};

module.exports = {
  get: get,
};

let extractTables = function(html) {
  xray(html, ['.table1.dividendhisttable'])(function(conversionError,
    tableHtmlList) {
    if (conversionError) {
      return reject(conversionError);
    }
    let tableList = tableHtmlList.map(function(table) {
      // xray returns the html inside each table tag, and tabletojson
      // expects a valid html table, so we need to re-wrap the table.
      // Returning the first element in the converted array because
      // we should only ever be parsing one table at a time within this map.
      return tabletojson.convert('<table>' + table + '</table>')[0];
    });

    tableList.forEach((table) => {
      console.log(table);
    });
  });
};

let extractHTML = function(filePath) {
  return new Promise(function(resolve, reject) {
    resolve(fs.readFileSync(filePath, 'utf8'));
  });
};

let runExtraction = function() {
  extractHTML('./html/test.htm').then((html) => {
    extractTables(html);
  })
    .catch((err) => {
      console.log(err);
    });
};

runNightmare();
