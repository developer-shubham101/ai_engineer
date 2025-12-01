# Authentication & Login Context

## JWT Authentication System

### Token Creation
- **Function**: `create_access_token(user_data, session_id=None)`
- **Expiration**: Configurable via `JWT_EXPIRATION_DAYS`
- **Algorithm**: Configurable via `JWT_ALGORITHM`
- **Secret**: Secured via `JWT_SECRET_KEY`

### Token Payload Structure
```json
{
  "user_id": "string",
  "username": "string", 
  "role": "string",
  "department": "string",
  "session_id": "string (optional)",
  "exp": "datetime",
  "iat": "datetime"
}
```

### Authentication Endpoint
- **URL**: `POST /api/auth/token`
- **Purpose**: JWT login for role-based access

### Role Hierarchy (RBAC)
- **SuperAdmin (4)** - Full system access
- **Manager (3)** - Management + below
- **HR (2)** - HR functions + below  
- **Employee (1)** - Standard access + public
- **Guest/PublicUser (0)** - Public content only

### Security Features
- JWT token verification with expiration handling
- User action logging for token creation
- Security event logging for invalid/expired tokens
- Sensitive debug logging (remove in production)

### Token Verification
- **Function**: `verify_token(token)`
- **Returns**: Decoded payload or None if invalid
- **Handles**: Expired tokens, invalid signatures, malformed tokens