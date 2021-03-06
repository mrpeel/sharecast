'use strict';


let max = function(array) {
  return Math.max.apply(null, array);
};

let min = function(array) {
  return Math.min.apply(null, array);
};

let range = function(array) {
  return max(array) - min(array);
};

let midrange = function(array) {
  return range(array) / 2;
};

let sum = function(array) {
  let num = 0;
  for (let i = 0, l = array.length; i < l; i++) num += array[i];
  return num;
};

let mean = function(array) {
  return sum(array) / array.length;
};

let average = function(array) {
  return mean(array);
};

let median = function(array) {
  array.sort(function(a, b) {
    return a - b;
  });

  let mid = array.length / 2;

  return mid % 1 ? array[mid - 0.5] : (array[mid - 1] + array[mid]) / 2;
};

let modes = function(array) {
  if (!array.length) return [];

  let modeMap = {};
  let maxCount = 1;
  let modes = [array[0]];

  array.forEach((val) => {
    if (!modeMap[val]) {
      modeMap[val] = 1;
    } else {
      modeMap[val]++;
    }

    if (modeMap[val] > maxCount) {
      modes = [val];
      maxCount = modeMap[val];
    } else if (modeMap[val] === maxCount) {
      modes.push(val);
      maxCount = modeMap[val];
    }
  });

  return modes;
};

let variance = function(array) {
  let meanVal = mean(array);

  return mean(array.map((num) => {
    return Math.pow(num - meanVal, 2);
  }));
};

let standardDeviation = function(array) {
  return Math.sqrt(variance(array));
};

let stdDev = function(array) {
  return standardDeviation(array);
};


let meanAbsoluteDeviation = function(array) {
  let meanVal = mean(array);

  return mean(array.map((num) => {
    return Math.abs(num - meanVal);
  }));
};

let zScores = function(array) {
  let meanVal = mean(array);

  let standardDeviation = standardDeviation(array);

  return array.map((num) => {
    return (num - meanVal) / standardDeviation;
  });
};


module.exports = {
  max: max,
  min: min,
  range: range,
  midrange: midrange,
  sum: sum,
  mean: mean,
  average: average,
  median: median,
  modes: modes,
  variance: variance,
  standardDeviation: standardDeviation,
  stdDev: stdDev,
  meanAbsoluteDeviation: meanAbsoluteDeviation,
  zScores: zScores,
};
