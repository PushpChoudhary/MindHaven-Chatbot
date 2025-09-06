// app/api/signup/route.js
import { NextResponse } from 'next/server';
import { readUsers, writeUsers } from '@/lib/db';
import { hashPassword } from '@/lib/auth';

export async function POST(request) {
  const { name, email, password } = await request.json();
  if (!name || !email || !password) {
    return NextResponse.json({ error: 'Missing fields' }, { status: 400 });
  }

  const users = readUsers();
  const existing = users.find(u => u.email === email);
  if (existing) {
    return NextResponse.json({ error: 'Email already exists' }, { status: 400 });
  }

  const hashedPassword = hashPassword(password);
  const newUser = { name, email, password: hashedPassword };
  users.push(newUser);
  writeUsers(users);

  return NextResponse.json({ message: 'User created successfully' });
}
