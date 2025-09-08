import React from 'react';

const Footer = () => {
  return (
    <footer className="bg-black text-white py-6 mt-10">
      <div className="container mx-auto text-center">
        <p className="text-sm">&copy; 2025 MindHaven. All Rights Reserved.</p>
        <div className="flex justify-center gap-6 mt-4">
          <a href="#" className="hover:text-indigo-400">Privacy Policy</a>
          <a href="#" className="hover:text-indigo-400">Terms of Service</a>
          <a href="#" className="hover:text-indigo-400">Contact Us</a>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
