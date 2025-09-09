'use client';

import Image from 'next/image';

export default function AboutPage() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-start p-8 bg-white text-gray-800">
      <h1 className="text-4xl font-bold text-indigo-600 mb-4">🕊️ About MindHaven</h1> {/* Changed MindHaven to MindHaven */}

      <div className="max-w-3xl text-center space-y-6 mb-12">
        <p className="text-lg">
          <strong>MindHaven</strong> is a friendly mental health chatbot designed to provide emotional support, {/* Changed MindHavenn to MindHaven */}
          guidance, and a listening ear to anyone in need. We aim to make mental wellness more accessible,
          compassionate, and stigma-free for everyone.
        </p>
        <p>
          Whether you're feeling anxious, overwhelmed, or just need someone to talk to, MindHaven is here to help {/* Changed MindHaven to MindHaven */}
          like a friend. Your messages are private and respected, and we strive to offer thoughtful responses
          powered by AI and psychology-inspired prompts.
        </p>
        <p>
          We believe that taking care of your mind is just as important as taking care of your body.
          That’s why we’re building a space where mental health support is available 24/7 — anytime, anywhere.
        </p>
        <p className="font-semibold text-indigo-700">
          You're not alone. We're here for you. 💙
        </p>
      </div>

      {/* Centering the heading */}
      <h2 className="text-3xl font-bold text-indigo-500 mb-6 text-center w-full">👩‍💻 Meet the Creators</h2>

      {/* Adjusting the grid for better alignment with two items */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-2xl mx-auto w-full justify-center"> {/* Changed max-w-5xl to max-w-2xl and added mx-auto, w-full, justify-center */}
        {/* Person 1: Chetan */}
        <div className="flex flex-col items-center bg-gray-100 p-6 rounded-2xl shadow">
          <Image
            src="/pushp.jpeg" // Place actual image in /public folder
            alt="Pushp Choudhary"
            width={150}
            height={150}
            className="rounded-full object-cover"
          />
          <h3 className="mt-4 text-xl font-semibold">Pushp Choudhary</h3>
          <p>College: NIET</p>
        </div>
        <div className="flex flex-col items-center bg-gray-100 p-6 rounded-2xl shadow">
          <Image
            src="/chetan.jpeg" // Place actual image in /public folder
            alt="Chetan Chauhan"
            width={150}
            height={150}
            className="rounded-full object-cover"
          />
          <h3 className="mt-4 text-xl font-semibold">Chetan Chauhan</h3>
          <p>College: NIET</p>
        </div>

        {/* Person 2: Princee */}
        <div className="flex flex-col items-center bg-gray-100 p-6 rounded-2xl shadow">
          <Image
            src="/Princee.jpeg"
            alt="Princee"
            width={150}
            height={10}
            className="rounded-full object-cover"
          />
          <h3 className="mt-4 text-xl font-semibold">Princee</h3>
          <p>College: NIET</p>
        </div>
      </div>
    </div>
  );
}
