import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

// Single stacked-pets image
import petStack from './img/stack.png';

axios.defaults.baseURL = ''; // CRA proxy → http://localhost:5000

export default function App() {
  const [date, setDate] = useState('');
  const [nextRefill, setNextRefill] = useState(null);
  const [interval, setInterval] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async e => {
    e.preventDefault();
    setError(null);
    setNextRefill(null);
    setInterval(null);

    if (!date) {
      setError('Please pick a date.');
      return;
    }

    try {
      const { data } = await axios.post('/predict', { last: date });
      setNextRefill(data.next_refill);
      setInterval(data.interval_hours);
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    }
  };

  return (
    <div className="App">
      <div className="left-panel">
        <h1>PawPlate</h1>
        <div className="image-stack">
          <img src={petStack} alt="Stacked pets" />
        </div>
      </div>
      <div className="right-panel">
        <blockquote className="quote">
          <span className="open-quote">“</span>
          <p>Feed ’Em & Fido Will Do the Rest</p>
          <span className="close-quote">”</span>
        </blockquote>

        <div className="form-container" style={{ border: '2px solid var(--text)' }}>
          <form onSubmit={handleSubmit}>
            <input
              type="date"
              value={date}
              onChange={e => setDate(e.target.value)}
              required
            />
            <button type="submit">find next refill date</button>
          </form>
        </div>
        {error && <div className="error">⚠️ {error}</div>}

        {nextRefill && (
          <div className="result">
            <p>🔄 Predicted next interval: <strong>{interval} hrs</strong></p>
            <p>🐾 Next refill: <strong>{nextRefill}</strong></p>
          </div>
        )}
      </div>
    </div>
  );
}