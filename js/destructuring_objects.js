const sandwich = {
  bread: "dutch crunch",
  meat: "tuna",
  cheese: "swiss",
  toppings: ["lettuce", "tomato", "mustard"]
};

const { bread, meat } = sandwich;

console.log(bread, meat); // dutch crunch tuna

// or you can use different names of variables

const { bread: myBread, meat: myMeat } = sandwich;

console.log(myBread, myMeat); // dutch crunch tuna

/*
When you use destructuring assignment in JavaScript to extract values from an object, it effectively makes a copy of the value for primitive data types (such as strings, numbers, and booleans) and assigns it to the new variable. This means that if you modify the new variable, the original value in the object remains unchanged because the value is copied.

However, for non-primitive values (such as objects and arrays), what is copied is a reference to the original object, not the actual object itself. This means that if the destructured variable is an object or an array and you modify it, you're modifying the original object or array because both the original property and the new variable refer to the same underlying data in memory.
*/

const person = {
  name: 'John',
  age: 30,
  hobbies: ['reading', 'cycling']
};

// Destructuring to copy primitive and non-primitive values
let { name, hobbies } = person;

// Modifying the primitive value does not affect the original object
name = 'Jane';
console.log(person.name); // Outputs: John

// Modifying the non-primitive value (array) affects the original object
hobbies.push('swimming');
console.log(person.hobbies); // Outputs: ['reading', 'cycling', 'swimming']
