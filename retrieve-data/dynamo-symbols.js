'use strict';

const dynamodb = require('./dynamodb');


let getCompanies = async function() {
  try {
    let companies = [];
    let getDetails = {
      tableName: 'companies',
    };

    let rows = await dynamodb.getTable(getDetails);

    rows.forEach((row) => {
      companies.push({
        'symbol': row.symbol,
        'yahoo-symbol': row.symbolYahoo,
      });
    });
    return companies;
  } catch (err) {
    console.log(err);
    throw new Error(err);
  }
};

let addCompany = async function(companyDetails) {
  if (!companyDetails.symbol) {
    console.error('companyDetails missing symbol: ' +
      JSON.stringify(companyDetails));
    return 'companyDetails missing symbol: ' +
      JSON.stringify(companyDetails);
  }

  try {
    if (!companyDetails.symbolYahoo) {
      companyDetails.symbolYahoo = companyDetails.symbol + '.AX';
    }

    if (!companyDetails.symbolGoogle) {
      companyDetails.symbolGoogle = 'ASX:' + companyDetails.symbol;
    }

    let insertDetails = {
      tableName: 'companies',
      values: companyDetails,
      primaryKey: ['symbol'],
    };

    await dynamodb.insertRecord(insertDetails);
  } catch (err) {
    console.error(err);
  }
};

let removeCompany = async function(companySymbol) {
  if (!companySymbol) {
    console.error('Missing symbol');
    return 'Missing symbol';
  }

  try {
    let deleteDetails = {
      tableName: 'companies',
      key: {
        symbol: companySymbol,
      },
    };

    await dynamodb.deleteRecord(deleteDetails);
  } catch (err) {
    console.error(err);
  }
};

let getIndices = async function() {
  try {
    let companies = [];
    let getDetails = {
      tableName: 'indices',
    };

    let rows = await dynamodb.getTable(getDetails);

    rows.forEach((row) => {
      companies.push({
        'symbol': row.symbol,
        'yahoo-symbol': row.symbolYahoo,
      });
    });
    return companies;
  } catch (err) {
    console.log(err);
    throw new Error(err);
  }
};


module.exports = {
  getCompanies: getCompanies,
  getIndices: getIndices,
  addCompany: addCompany,
  removeCompany: removeCompany,
};
