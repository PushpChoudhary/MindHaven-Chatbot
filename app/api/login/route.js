
import { NextResponse } from 'next/server';
import clientPromise from '@/lib/mongodb';
import { comparePassword, generateToken } from '@/lib/auth';
import { serialize } from 'cookie';

export async function POST(request) {
  try {
    const { email, password } = await request.json();
    if (!email || !password) {
      return NextResponse.json({ error: 'Missing email or password' }, { status: 400 });
    }

    const client = await clientPromise;
    const db = client.db();
    const usersCollection = db.collection('users');

    // Find the user by email
    const user = await usersCollection.findOne({ email });

    // Check if user exists and password is correct
    if (!user || !comparePassword(password, user.password)) {
      return NextResponse.json({ error: 'Invalid credentials' }, { status: 401 });
    }
    
    // User is valid, generate a token
    const token = generateToken(user);
    
    // Return a success response and set the token in a cookie
    const response = NextResponse.json({ message: 'Login successful' }, { status: 200 });
    
    response.headers.set('Set-Cookie', serialize('token', token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      path: '/',
      maxAge: 60 * 60 * 24 * 7, // 1 week
    }));

    return response;
  } catch (error) {
    console.error('Login Error:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
