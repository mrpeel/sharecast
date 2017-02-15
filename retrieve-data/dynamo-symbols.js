const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');
const dynamodb = require('./dynamodb');


let getCompanies = asyncify(function() {
  return new Promise(function(resolve, reject) {
    try {
      let companies = [];
      let getDetails = {
        tableName: 'companies',
      };

      let rows = awaitify(dynamodb.getTable(getDetails));

      rows.forEach((row) => {
        companies.push({
          'symbol': row.symbol,
          'yahoo-symbol': row.symbolYahoo,
        });
      });
      resolve(companies);
    } catch (err) {
      console.log(err);
      reject(err);
    }
  });
});

let addCompany = asyncify(function(companyDetails) {
  if (!companyDetails.symbol) {
    reject('companyDetails missing symbol: ' +
      JSON.stringify(companyDetails));
  }

  try {
    if (!companyDetails.symbolYahoo) {
      companyDetails.symbolYahoo = companyDetails.symbol + '.AX';
    }

    if (!companyDetails.symbolGoogle) {
      companyDetails.symbolGoogle = companyDetails.symbol + ':AX';
    }

    let insertDetails = {
      tableName: 'companies',
      values: companyDetails,
      primaryKey: ['symbol'],
    };

    awaitify(dynamodb.insertRecord(insertDetails));
  } catch (err) {
    console.log(err);
  }
});

let getIndices = asyncify(function() {
  return new Promise(function(resolve, reject) {
    try {
      let companies = [];
      let getDetails = {
        tableName: 'indices',
      };

      let rows = awaitify(dynamodb.getTable(getDetails));

      rows.forEach((row) => {
        companies.push({
          'symbol': row.symbol,
          'yahoo-symbol': row.symbolYahoo,
        });
      });
      resolve(companies);
    } catch (err) {
      console.log(err);
      reject(err);
    }
  });
});

/* let migrateData = asyncify(function() {
  symbols.forEach((symbolVal) => {
    let companyDetails = {
      symbol: symbolVal.CompanySymbol,
      companyName: symbolVal.CompanyName,
      symbolYahoo: symbolVal.CompanySymbolYahoo,
      symbolGoogle: symbolVal.CompanySymbolGoogle,
    };
    try {
      awaitify(addCompany(companyDetails));
    } catch (err) {
      console.log(err);
    }
  });
}); */

/* let testRetrieve = asyncify(function() {
  let symbols = awaitify(getIndices());
  console.log(symbols);
  symbols = awaitify(getCompanies());
  console.log(symbols);
});

testRetrieve();*/

module.exports = {
  getCompanies: getCompanies,
  getIndices: getIndices,
  addCompany: addCompany,
};
