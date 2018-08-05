'use strict';

const json2csv = require('json2csv');
const fs = require('fs');
const moment = require('moment-timezone');

let sleep = function(ms) {
  if (!ms) {
    ms = 1;
  }
  return new Promise((r) => {
    setTimeout(r, ms);
  });
};


/** Sets up or returns the duration of an activity
 * @param {Object} start - optional - for the end of an activity, the start
 *                          object to compare with
 * @return {Object} if no start is supplied, returns a start object; if startRec
 *                   is supplied, returns a duration object
 */
let getTiming = function(start) {
  if (!start) {
    return process.hrtime();
  }
  let end = process.hrtime(start);
  return end[0] + (end[1] / 1000000000);
};


/** Returns a date in a set format: YYYY-MM-DD
 * @param {String} dateValue - the date string to parse
 * @param {String} dateFormat - a string defining the input format:
 *  YYYY 2014 4 or 2 digit year
 *  YY 14 2 digit year
 *  Y 25 Year with any number of digits and sign
 *  Q 1..4 Quarter of year. Sets month to first month in quarter.
 *  M MM 1..12 Month number
 *  MMM MMMM Jan..December Month name in locale set by moment.locale()
 *  D DD 1..31 Day of month
 *  Do 1st..31st Day of month with ordinal
 *  DDD DDDD 1..365 Day of year
 *  X 1410715640.579 Unix timestamp
 *  x 1410715640579 Unix ms timestamp
 * @return {String} the reformatted date or an empty string
 */
let returnDateAsString = function(dateValue, dateFormat) {
  if (moment(dateValue, dateFormat || '').isValid()) {
    return moment(dateValue, dateFormat || '').format('YYYY-MM-DD');
  } else {
    return '';
  }
};

/** Returns a date as a unix timestamp
 * @param {String} dateValue - the date string to parse
 * @param {String} dateFormat - a string defining the input format:
 *  YYYY 2014 4 or 2 digit year
 *  YY 14 2 digit year
 *  Y 25 Year with any number of digits and sign
 *  Q 1..4 Quarter of year. Sets month to first month in quarter.
 *  M MM 1..12 Month number
 *  MMM MMMM Jan..December Month name in locale set by moment.locale()
 *  D DD 1..31 Day of month
 *  Do 1st..31st Day of month with ordinal
 *  DDD DDDD 1..365 Day of year
 *  X 1410715640.579 Unix timestamp
 *  x 1410715640579 Unix ms timestamp
 * @return {Float} the unix timestamp
 */
let returnDateAsUnix = function(dateValue, dateFormat) {
  if (moment(dateValue, dateFormat || '').isValid()) {
    return moment(dateValue, dateFormat || '').unix();
  } else {
    return -1;
  }
};


/** Returns a date for the end of the month in a set format: YYYY-MM-DD
 * @param {String} dateValue - the date string to parse
 * @param {String} dateFormat - a string defining the input format:
 *  YYYY 2014 4 or 2 digit year
 *  YY 14 2 digit year
 *  Y 25 Year with any number of digits and sign
 *  Q 1..4 Quarter of year. Sets month to first month in quarter.
 *  M MM 1..12 Month number
 *  MMM MMMM Jan..December Month name in locale set by moment.locale()
 *  D DD 1..31 Day of month
 *  Do 1st..31st Day of month with ordinal
 *  DDD DDDD 1..365 Day of year
 *  X 1410715640.579 Unix timestamp
 *  x 1410715640579 Unix ms timestamp
 * @return {String} the formatted end of month date or an empty string
 */
let returnEndOfMonth = function(dateValue, dateFormat) {
  if (moment(dateValue, dateFormat || '').isValid()) {
    return moment(dateValue, dateFormat || '')
      .endOf('month')
      .format('YYYY-MM-DD');
  } else {
    return '';
  }
};

/**
 * converts the date string used for querying to a formated date string which
 *  can be displayed
 * @param {String} dateValue1 a date or string in a format which can be
 *   converted to a date
 * @param {String} dateValue2 a date or string in a format which can be
 *   converted to a date
 * @param {string} unit the unit to change by "seconds", "minutes", "hours",
 "days" , "weeks", "months", "years"
 * @return {Number} the difference as a number
 */
let dateDiff = function(dateValue1, dateValue2, unit) {
  // Check that this really is a date
  if (!moment(dateValue1).isValid()) {
    console.error('dateValue1 invalid: ' + dateValue1);
    return null;
  }

  if (!moment(dateValue2).isValid()) {
    console.error('dateValue2 invalid: ' + dateValue2);
    return null;
  }

  if (!(unit === 'milliseconds' || unit === 'seconds' || unit === 'minutes' ||
      unit === 'hours' || unit === 'days' || unit === 'weeks' ||
      unit === 'months' || unit === 'years')) {
    console.error('unit invalid: ' + unit);
    return null;
  }

  let md1 = moment(dateValue1);
  let md2 = moment(dateValue2);

  if (unit === 'milliseconds') {
    return Math.abs(md1.diff(md2));
  } else {
    return Math.abs(md1.diff(md2, unit));
  }
};


/**
 * converts the date string used for querying to a formated date string which
 *  can be displayed
 * @param {String} dateValue a date or string in a format which can be
 *   converted to a date
 * @param {string} unit the unit to change by "days" days, "weeks" weeks,
 *                    "months", "year" years
 * @param {number} number to change, positive number for futures, negative
 *    number for past
 * @return {Date} a date with the new value
 */
let dateAdd = function(dateValue, unit, number) {
  // Check that this really is a date
  if (!moment(dateValue).isValid()) {
    console.error('dateValue invalid: ' + dateValue);
    return null;
  }
  if (!(unit === 'days' || unit === 'weeks' || unit === 'months' ||
      unit === 'years')) {
    console.error('unit invalid: ' + unit);
    return null;
  }
  if (typeof number !== 'number') {
    console.error('number invalid: ' + number);
    return null;
  }

  return moment(dateValue).add(number, unit).format('YYYY-MM-DD');
};

let isDate = function(dateValue, dateFormat) {
  // Check that this really is a date
  return moment(dateValue, dateFormat || '').isValid();
};

let createFieldArray = function(fieldObject) {
  return Object.keys(fieldObject);
};

let writeToCsv = function(csvData, csvFields, filePrefix, dateString) {
  // If no date string supplied, use today's date
  let today = new Date();

  dateString = dateString || today.getFullYear() + '-' +
    ('0' + (today.getMonth() + 1)).slice(-2) + '-' +
    ('0' + today.getDate()).slice(-2);

  let csvOutput = json2csv({
    data: csvData,
    fields: csvFields,
  });

  fs.writeFile('../data/' + filePrefix + '-' + dateString + '.csv',
    csvOutput,
    function(err) {
      if (err) {
        throw err;
      }
      console.log('File saved');
    });
};

let writeJSONfile = function(jsonObject, fileName) {
  fs.writeFile(fileName, JSON.stringify(jsonObject), function(err) {
    if (err) {
      throw err;
    }
    console.log('JSON File saved');
  });
};

let doesDataFileExist = function(fileName) {
  return fs.existsSync('../data/' + fileName);
};

let checkForNumber = function(value) {
  let finalChar = String(value).slice(-1);
  let leadingVal = String(value).slice(0, String(value).length - 1);
  let adjustedVal = value;

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
    adjustedVal = leadingVal * charsToMatch[finalChar];
  }

  // If it's not a number, replace commas in value, then check if it's a number
  if (isNaN(adjustedVal)) {
    let holdingVal = adjustedVal.replace(/,/g, '');

    if (!isNaN(holdingVal)) {
      adjustedVal = holdingVal;
    }
  }
  return adjustedVal;
};


module.exports = {
  sleep: sleep,
  getTiming: getTiming,
  returnDateAsString: returnDateAsString,
  returnDateAsUnix: returnDateAsUnix,
  dateAdd: dateAdd,
  dateDiff: dateDiff,
  isDate: isDate,
  createFieldArray: createFieldArray,
  writeToCsv: writeToCsv,
  writeJSONfile: writeJSONfile,
  doesDataFileExist: doesDataFileExist,
  checkForNumber: checkForNumber,
  returnEndOfMonth: returnEndOfMonth,
};
