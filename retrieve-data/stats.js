
let max = function(array) {
  return Math.max.apply(null, array);
};

let min = function(array) {
  return Math.min.apply(null, array);
};

let range = function(array) {
  return arr.max(array) - arr.min(array);
};

let midrange = function(array) {
  return arr.range(array) / 2;
};

let sum = function(array) {
  let num = 0;
  for (let i = 0, l = array.length; i < l; i++) num += array[i];
  return num;
};

let mean = function(array) {
  return arr.sum(array) / array.length;
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
  let mean = arr.mean(array);

  return arr.mean(array.map((num) => {
    return Math.pow(num - mean, 2);
  }));
};

let standardDeviation = function(array) {
  return Math.sqrt(arr.variance(array));
};

let meanAbsoluteDeviation = function(array) {
  let mean = arr.mean(array);

  return arr.mean(array.map((num) => {
    return Math.abs(num - mean);
  }));
};

let zScores = function(array) {
  let mean = arr.mean(array);

  let standardDeviation = arr.standardDeviation(array);

  return array.map((num) => {
    return (num - mean) / standardDeviation;
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
  meanAbsoluteDeviation: meanAbsoluteDeviation,
  zScores: zScores,
};
