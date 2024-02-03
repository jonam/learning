/*
Slice method seems to be fastest.

Create array: 21.214s
Spread Operator: 171.514ms
Slice Method: 116.88ms
Array.from: 355.532ms
*/

console.time('Create array');
const largeArray = new Array(100000000).fill(0).map((_, i) => i);
console.timeEnd('Create array');

console.time('Spread Operator');
const copiedArray1 = [...largeArray];
console.timeEnd('Spread Operator');

console.time('Slice Method');
const copiedArray2 = largeArray.slice();
console.timeEnd('Slice Method');

console.time('Array.from');
const copiedArray3 = Array.from(largeArray);
console.timeEnd('Array.from');
