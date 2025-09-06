// lib/db.js
import fs from 'fs';
import path from 'path';

const DB_PATH = path.resolve(process.cwd(), 'users.json');

export function readUsers() {
  if (!fs.existsSync(DB_PATH)) return [];
  const data = fs.readFileSync(DB_PATH);
  return JSON.parse(data);
}

export function writeUsers(users) {
  fs.writeFileSync(DB_PATH, JSON.stringify(users, null, 2));
}
