/*
To make a copy of non-primitive values in JavaScript, such as objects and arrays, you need to ensure that the copy is done in a way that doesn't just copy the reference to the original object/array, but instead creates a new object/array with the same properties/values. Here are common ways to do this:

For Arrays
Using the Spread Operator (...):
It's syntactically concise and widely used for its readability and ease of use. Performance-wise, it's generally efficient for small to moderate-sized arrays.
*/

const originalArray = [1, 2, 3];
const copiedArray = [...originalArray];

// Using the slice() method:
/*
Historically, slice() has been a go-to method for copying arrays due to its straightforward semantics. It has been highly optimized in many JavaScript engines, making it very efficient for copying arrays.
*/

const originalArray = [1, 2, 3];
const copiedArray = originalArray.slice();

// Using Array.from():
/*
This method is versatile, as it can create new arrays from array-like or iterable objects. While it is also efficient, its performance might be slightly less optimized for simple array copying scenarios compared to slice(), simply because it's designed for a broader set of use cases.
*/

const originalArray = [1, 2, 3];
const copiedArray = Array.from(originalArray);

/*
For Objects
Using the Spread Operator (...):
*/

const originalObject = { a: 1, b: 2 };
const copiedObject = { ...originalObject };

// Using Object.assign():

const originalObject = { a: 1, b: 2 };
const copiedObject = Object.assign({}, originalObject);

/*
Deep Copy for Nested Objects/Arrays
The methods mentioned above perform a shallow copy, which is fine for top-level properties. However, if the object or array contains nested objects or arrays, these nested elements will still be copied by reference. To perform a deep copy, where nested objects/arrays are also copied by value, you can use:

JSON Serialization:
This method can be used for deep copying but has limitations, such as not copying functions, circular references, or special objects like Date, RegExp, etc.
*/

const originalObject = { a: 1, b: { c: 2 } };
const copiedObject = JSON.parse(JSON.stringify(originalObject));

/*
Libraries for Deep Copy:
Libraries like Lodash offer deep copy functions (_.cloneDeep()) that handle various edge cases and data types more gracefully than the JSON serialize/deserialize method.
*/

const copiedObject = _.cloneDeep(originalObject);

/*
Note that you would need to include Lodash in your project to use this method.

It's important to choose the right method based on whether you need a shallow or deep copy and depending on the complexity of the objects or arrays you're working with.
*/
