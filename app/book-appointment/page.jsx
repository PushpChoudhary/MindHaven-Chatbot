
'use client';

import { useState } from 'react';

export default function BookAppointmentPage() {
  const [form, setForm] = useState({
    name: '',
    email: '',
    datetime: '',
    message: '',
  });
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState('');

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async () => {
    const { name, email, datetime } = form;

    if (!name || !email || !datetime) {
      setResponse('Please fill in all required fields.');
      return;
    }

    setLoading(true);
    setResponse('');

    try {
      // ✅ FIX: Use backticks (`) instead of single quotes (')
      const res = await fetch(`${process.env.NEXT_PUBLIC_PYTHON_API_URL}/book-appointment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });

      const data = await res.json();
      setResponse(data.message || data.error || 'No response from server');
    } catch (error) {
      setResponse('Failed to connect to backend');
    }

    setLoading(false);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-start p-6 bg-white">
      <h1 className="text-3xl font-bold mb-6 text-indigo-600">📅 Book an Appointment</h1>

      <div className="w-full max-w-xl space-y-4">
        <input
          type="text"
          name="name"
          placeholder="Your Name"
          className="w-full p-3 border border-gray-300 rounded-lg shadow-sm"
          value={form.name}
          onChange={handleChange}
        />
        <input
          type="email"
          name="email"
          placeholder="Your Email"
          className="w-full p-3 border border-gray-300 rounded-lg shadow-sm"
          value={form.email}
          onChange={handleChange}
        />
        <input
          type="datetime-local"
          name="datetime"
          className="w-full p-3 border border-gray-300 rounded-lg shadow-sm"
          value={form.datetime}
          onChange={handleChange}
        />
        <textarea
          name="message"
          rows={4}
          placeholder="Any message or concern..."
          className="w-full p-3 border border-gray-300 rounded-lg shadow-sm"
          value={form.message}
          onChange={handleChange}
        />

        <button
          className="bg-indigo-600 text-white px-6 py-2 rounded hover:bg-indigo-700 transition w-full"
          onClick={handleSubmit}
          disabled={loading}
        >
          {loading ? 'Booking...' : 'Book Appointment'}
        </button>

        {response && (
          <div className="mt-4 p-4 bg-gray-100 border border-gray-300 rounded">
            <strong className="text-gray-800">Response:</strong>
            <p className="mt-2 text-gray-700 whitespace-pre-wrap">{response}</p>
          </div>
        )}
      </div>
    </div>
  );
}

