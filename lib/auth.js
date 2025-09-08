
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';

const JWT_SECRET = process.env.JWT_SECRET;

export function hashPassword(password) {
  // A salt is random data that is used as an additional input to a one-way function that hashes data.
  const salt = bcrypt.genSaltSync(10);
  return bcrypt.hashSync(password, salt);
}

export function comparePassword(password, hashedPassword) {
  return bcrypt.compareSync(password, hashedPassword);
}

export function generateToken(user) {
  // The token will contain the user's ID and email, and it will expire in 7 days.
  const payload = {
    id: user._id, // MongoDB uses _id as the primary key
    email: user.email,
  };
  return jwt.sign(payload, JWT_SECRET, { expiresIn: '7d' });
}
