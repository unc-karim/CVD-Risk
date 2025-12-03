/**
 * Persistent navigation bar accessible from all pages
 */

import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';
import HomeIcon from '@mui/icons-material/Home';
import FavoriteBorderIcon from '@mui/icons-material/FavoriteBorder';
import TimelineIcon from '@mui/icons-material/Timeline';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import HubIcon from '@mui/icons-material/Hub';
import InfoIcon from '@mui/icons-material/Info';

const Navigation: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const isActive = (path: string) => location.pathname === path;

  return (
    <AppBar position="sticky" sx={{ bgcolor: 'white', color: '#1F2937', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08)' }}>
      <Toolbar sx={{ justifyContent: 'space-between' }}>
        {/* Logo */}
        <Box
          onClick={() => navigate('/')}
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 1,
            cursor: 'pointer',
            textDecoration: 'none',
          }}
        >
          <Typography
            variant="h6"
            sx={{
              fontWeight: 700,
              color: '#5939E0',
              fontSize: '1.1rem',
              letterSpacing: '-0.3px',
            }}
          >
            CVD Risk
          </Typography>
        </Box>

        {/* Navigation Links */}
        <Box sx={{ display: 'flex', gap: 0.5 }}>
          <Button
            onClick={() => navigate('/')}
            startIcon={<HomeIcon />}
            sx={{
              textTransform: 'none',
              color: isActive('/') ? '#5939E0' : '#6B7280',
              fontWeight: isActive('/') ? 700 : 600,
              fontSize: '0.9rem',
              '&:hover': {
                backgroundColor: 'rgba(89, 57, 224, 0.08)',
                color: '#5939E0',
              },
            }}
          >
            Home
          </Button>

          <Button
            onClick={() => navigate('/htn')}
            startIcon={<FavoriteBorderIcon />}
            sx={{
              textTransform: 'none',
              color: isActive('/htn') ? '#5939E0' : '#6B7280',
              fontWeight: isActive('/htn') ? 700 : 600,
              fontSize: '0.9rem',
              '&:hover': {
                backgroundColor: 'rgba(89, 57, 224, 0.08)',
                color: '#5939E0',
              },
            }}
          >
            HTN
          </Button>

          <Button
            onClick={() => navigate('/cimt')}
            startIcon={<TimelineIcon />}
            sx={{
              textTransform: 'none',
              color: isActive('/cimt') ? '#5939E0' : '#6B7280',
              fontWeight: isActive('/cimt') ? 700 : 600,
              fontSize: '0.9rem',
              '&:hover': {
                backgroundColor: 'rgba(89, 57, 224, 0.08)',
                color: '#5939E0',
              },
            }}
          >
            CIMT
          </Button>

          <Button
            onClick={() => navigate('/vessel')}
            startIcon={<AccountTreeIcon />}
            sx={{
              textTransform: 'none',
              color: isActive('/vessel') ? '#5939E0' : '#6B7280',
              fontWeight: isActive('/vessel') ? 700 : 600,
              fontSize: '0.9rem',
              '&:hover': {
                backgroundColor: 'rgba(89, 57, 224, 0.08)',
                color: '#5939E0',
              },
            }}
          >
            Vessels
          </Button>

          <Button
            onClick={() => navigate('/fusion')}
            startIcon={<HubIcon />}
            sx={{
              textTransform: 'none',
              color: isActive('/fusion') ? '#5939E0' : '#6B7280',
              fontWeight: isActive('/fusion') ? 700 : 600,
              fontSize: '0.9rem',
              '&:hover': {
                backgroundColor: 'rgba(89, 57, 224, 0.08)',
                color: '#5939E0',
              },
            }}
          >
            Fusion
          </Button>

          <Button
            onClick={() => navigate('/about')}
            startIcon={<InfoIcon />}
            sx={{
              textTransform: 'none',
              color: isActive('/about') ? '#5939E0' : '#6B7280',
              fontWeight: isActive('/about') ? 700 : 600,
              fontSize: '0.9rem',
              '&:hover': {
                backgroundColor: 'rgba(89, 57, 224, 0.08)',
                color: '#5939E0',
              },
            }}
          >
            About
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navigation;
