'use client';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Banner from "../_components/Home/Banner";
import Footer from "../_components/Home/Footer";

export default function Home() {
  const router = useRouter();
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) {
      router.push('/login');
    } else {
      setIsAuthenticated(true);
    }
  }, [router]);

  if (!isAuthenticated) {
    return <div className="text-center mt-20">Checking authentication...</div>;
  }

  return (
    <div className="container mx-auto p-6 lg:px-8">
      <Banner />
      <Footer />
    </div>
  );
}
