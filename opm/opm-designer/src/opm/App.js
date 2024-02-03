// App.js
import React from 'react';
import Toolbar from './components/Toolbar';
import Sidebar from './components/Sidebar';
import CanvasArea from './components/CanvasArea';
import StatusBar from './components/StatusBar';
import './App.css';

function App() {
  return (
    <div className="App">
      <Toolbar />
      <div className="MainArea">
        <Sidebar />
        <CanvasArea />
      </div>
      <StatusBar />
    </div>
  );
}

export default App;
