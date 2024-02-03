import React, { useEffect, useState } from 'react';
import './App.css';

function App() {
  const [message, setMessage] = useState('');

  useEffect(() => {
    // Fetch the hello world from the Flask server
    fetch('http://127.0.0.1:5000/')
      .then(response => response.text())
      .then(data => {
        setMessage(data);
      })
      .catch((error) => {
        console.error('Error fetching data: ', error);
        setMessage('Error fetching data');
      });
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <p>
          {message || "Loading..."}
        </p>
      </header>
    </div>
  );
}

export default App;

