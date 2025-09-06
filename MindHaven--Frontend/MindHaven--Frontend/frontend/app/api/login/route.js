// app/api/login/route.js
import { NextResponse } from 'next/server';
import { readUsers } from '@/lib/db';
import { comparePassword, generateToken } from '@/lib/auth';
import { serialize } from 'cookie';  // ✅ FIXED

export async function POST(request) {
  const { email, password } = await request.json();
  if (!email || !password) {
    return NextResponse.json({ error: 'Missing email or password' }, { status: 400 });
  }

  const users = readUsers();
  const user = users.find(u => u.email === email);
  if (!user || !comparePassword(password, user.password)) {
    return NextResponse.json({ error: 'Invalid credentials' }, { status: 401 });
  }

  const token = generateToken(user);
  const res = NextResponse.json({ message: 'Login successful' });

  // ✅ Serialize cookie correctly
  res.headers.set('Set-Cookie', serialize('token', token, {
    httpOnly: true,
    maxAge: 60 * 60 * 24 * 7,
    path: '/',
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
  }));

  return res;
}
