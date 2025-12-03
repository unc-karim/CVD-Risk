/**
 * CIMT regression page
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  Card,
  CircularProgress,

  Grid,
  Paper,
  TextField,
  Typography,
  RadioGroup,
  FormControlLabel,
  Radio,
  Grow,
} from '@mui/material';
import BarChartIcon from '@mui/icons-material/BarChart';
import StraightIcon from '@mui/icons-material/Straight';
import ImageUpload from '../components/ImageUpload';
import ExportButtons from '../components/ExportButtons';
import api, { CIMTResult } from '../services/api';

interface CIMTPageProps {
  apiReady: boolean;
}

const CIMTPage: React.FC<CIMTPageProps> = ({ apiReady }) => {
  const navigate = useNavigate();
  const [leftImage, setLeftImage] = useState<File | null>(null);
  const [rightImage, setRightImage] = useState<File | null>(null);
  const [age, setAge] = useState(65);
  const [gender, setGender] = useState(1);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CIMTResult | null>(null);
  const [error, setError] = useState('');

  const handleClearLeftImage = () => {
    setLeftImage(null);
  };

  const handleClearRightImage = () => {
    setRightImage(null);
  };

  const handleClearAll = () => {
    if (leftImage) {
      try { const u = URL.createObjectURL(leftImage); if (u.startsWith('blob:')) URL.revokeObjectURL(u); } catch (e) {}
    }
    if (rightImage) {
      try { const u = URL.createObjectURL(rightImage); if (u.startsWith('blob:')) URL.revokeObjectURL(u); } catch (e) {}
    }
    setLeftImage(null);
    setRightImage(null);
    setResult(null);
    setError('');
  };

  const handlePredict = async () => {
    if (!leftImage) {
      setError('Please select at least left eye image');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await api.predictCIMT(leftImage, rightImage, age, gender);
      if (response.status === 'success' && response.result) {
        setResult(response.result);
      } else {
        setError(response.error || 'Prediction failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ width: '100%', minHeight: '100vh', background: '#ffffff', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box
        sx={{
          background: 'linear-gradient(135deg, #5939E0 0%, #9D7FFF 100%)',
          color: 'white',
          py: { xs: 3, sm: 4, md: 5 },
          px: { xs: 2, sm: 3, md: 4 },
          mb: { xs: 4, md: 6 },
          borderBottomLeftRadius: { xs: 16, sm: 24, md: 32 },
          borderBottomRightRadius: { xs: 16, sm: 24, md: 32 },
          boxShadow: '0 8px 32px rgba(106, 77, 245, 0.25)',
        }}
      >
        <Box sx={{ maxWidth: 1200, mx: 'auto', display: 'flex', flexDirection: { xs: 'column', sm: 'row' }, alignItems: { xs: 'flex-start', sm: 'center' }, justifyContent: 'space-between', gap: { xs: 3, sm: 2, md: 4 } }}>
          <Box sx={{ flex: 1 }}>
            <Typography
              component="h1"
              variant="h2"
              sx={{
                mb: 1,
              }}
            >
              CIMT Prediction
            </Typography>
            <Typography variant="body1" sx={{ opacity: 0.85 }}>
              Carotid intima-media thickness estimation from retinal and clinical data
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', gap: 1.5, flexShrink: 0, flexWrap: 'wrap', justifyContent: { xs: 'flex-start', sm: 'flex-end' } }}>
            <Button
              onClick={() => navigate('/')}
              disabled={loading}
              variant="text"
              sx={{
                color: 'white',
                fontWeight: 600,
                fontSize: '0.95rem',
                py: 1,
                px: 2,
                '&:hover': {
                  backgroundColor: 'rgba(255,255,255,0.1)',
                },
              }}
            >
              ← Home
            </Button>
            <Button
              onClick={handleClearAll}
              disabled={loading}
              variant="outlined"
              sx={{
                color: 'white',
                borderColor: 'white',
                fontWeight: 600,
                fontSize: '0.95rem',
                py: 1,
                px: 3,
                transition: 'all 0.15s ease',
                '&:hover': {
                  backgroundColor: 'rgba(255,255,255,0.1)',
                  borderColor: 'white',
                },
              }}
            >
              Clear All
            </Button>
          </Box>
        </Box>
      </Box>
      <Box sx={{ py: 6, width: '100%' }}>
        <Grid container spacing={3} sx={{ width: '100%', maxWidth: '1200px', mx: 'auto', px: { xs: 2, sm: 3, md: 4 } }}>
          <Grid item xs={12}>
            <Paper sx={{ p: 3, backgroundColor: '#f8faff', maxWidth: 700, mx: 'auto' }}>
              <Typography variant="h6" sx={{ mb: 3, fontWeight: 600, color: '#0066cc' }}>
                Patient Information & Images
              </Typography>

              {/* Clinical Data - FIRST */}
              <Box sx={{ p: 2.5, backgroundColor: '#ffffff', borderRadius: 1, border: '1px solid #e0e0e0', mb: 3 }}>
                <Typography variant="subtitle2" sx={{ mb: 2.5, fontWeight: 600, color: '#555' }}>
                  Clinical Data
                </Typography>
                <TextField
                  type="number"
                  label="Age (years)"
                  fullWidth
                  value={age}
                  onChange={(e) => setAge(parseInt(e.target.value))}
                  inputProps={{ min: 1, max: 150 }}
                  variant="outlined"
                  size="small"
                  sx={{ mb: 2 }}
                />
                <Box sx={{ mb: 0 }}>
                  <Typography variant="body2" sx={{ mb: 1, fontWeight: 500, color: '#555' }}>
                    Sex
                  </Typography>
                  <RadioGroup value={gender.toString()} onChange={(e) => setGender(parseInt(e.target.value))} row>
                    <FormControlLabel value="0" control={<Radio size="small" />} label="Female" />
                    <FormControlLabel value="1" control={<Radio size="small" />} label="Male" />
                  </RadioGroup>
                </Box>
              </Box>

              {/* Retinal Images - now full width */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 600, color: '#555' }}>
                  Fundus Images
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <Box sx={{ width: '100%' }}>
                      <ImageUpload
                        onImageSelect={setLeftImage}
                        label="Left Eye Image (Required)"
                        imageUrl={leftImage ? URL.createObjectURL(leftImage) : ''}
                      />
                      {leftImage && (
                        <Button
                          size="small"
                          variant="outlined"
                          onClick={handleClearLeftImage}
                          disabled={loading}
                          sx={{
                            mt: 1,
                            width: '100%',
                            color: '#ef4444',
                            borderColor: '#ef4444',
                            '&:hover': {
                              backgroundColor: '#fee2e2',
                              borderColor: '#dc2626',
                            },
                          }}
                        >
                          Clear Image
                        </Button>
                      )}
                    </Box>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Box sx={{ width: '100%' }}>
                      <ImageUpload
                        onImageSelect={setRightImage}
                        label="Right Eye Image (Optional)"
                        imageUrl={rightImage ? URL.createObjectURL(rightImage) : ''}
                      />
                      {rightImage && (
                        <Button
                          size="small"
                          variant="outlined"
                          onClick={handleClearRightImage}
                          disabled={loading}
                          sx={{
                            mt: 1,
                            width: '100%',
                            color: '#ef4444',
                            borderColor: '#ef4444',
                            '&:hover': {
                              backgroundColor: '#fee2e2',
                              borderColor: '#dc2626',
                            },
                          }}
                        >
                          Clear Image
                        </Button>
                      )}
                    </Box>
                  </Grid>
                </Grid>
              </Box>

              <Button
                variant="contained"
                fullWidth
                onClick={handlePredict}
                disabled={!leftImage || loading || !apiReady}
                size="large"
                sx={{
                  mt: 3,
                  fontWeight: 700,
                  fontSize: '1.1rem',
                  py: 2,
                  backgroundColor: '#5939E0',
                  boxShadow: '0 2px 8px rgba(89, 57, 224, 0.2)',
                  '&:hover': {
                    backgroundColor: '#4A2DB0',
                    boxShadow: '0 4px 12px rgba(89, 57, 224, 0.25)',
                  },
                }}
              >
                {loading ? (
                  <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center' }}>
                    <CircularProgress size={20} color="inherit" />
                    Calculating...
                  </Box>
                ) : (
                  <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                    <StraightIcon sx={{ fontSize: '1.2rem' }} />
                    Estimate CIMT
                  </Box>
                )}
              </Button>

              {error && (
                <Paper sx={{ mt: 2, p: 2, backgroundColor: '#ffebee', border: '1px solid #ef5350', borderRadius: 1 }}>
                  <Typography variant="body2" sx={{ color: '#c62828', fontWeight: 500 }}>
                    {error}
                  </Typography>
                </Paper>
              )}

              {/* Results Section - now below input */}
              {result && (
                <Grow in={true} timeout={500}>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 4 }}>
                    {/* Main Result Card */}
                    <Card
                      sx={{
                        p: 4,
                        backgroundColor: '#ffffff',
                        borderLeft: `5px solid ${
                          result.risk_category === 'Elevated'
                            ? '#ef4444'
                            : result.risk_category === 'Borderline'
                            ? '#f59e0b'
                            : '#10b981'
                        }`,
                        borderRadius: 1,
                        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08)',
                      }}
                    >
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 1 }}>
                        <BarChartIcon sx={{ fontSize: '1rem', color: '#666' }} />
                        <Typography variant="overline" sx={{ fontWeight: 700, color: '#666', letterSpacing: 1 }}>
                          CIMT Measurement
                        </Typography>
                      </Box>
                      <Typography
                        variant="h2"
                        sx={{
                          fontWeight: 900,
                          mb: 1,
                          fontSize: '2.5rem',
                          color:
                            result.risk_category === 'Elevated'
                              ? '#ef4444'
                              : result.risk_category === 'Borderline'
                              ? '#f59e0b'
                              : '#10b981',
                        }}
                      >
                        {result.value_mm.toFixed(3)} mm
                      </Typography>

                      <Box sx={{ p: 2, backgroundColor: 'rgba(255, 255, 255, 0.8)', borderRadius: 1.5, border: '1px solid rgba(0,0,0,0.05)' }}>
                        <Typography variant="body2" sx={{ fontWeight: 500, color: '#666', mb: 0.8 }}>
                          Risk Category
                        </Typography>
                        <Typography
                          variant="h6"
                          sx={{
                            fontWeight: 700,
                            fontSize: '1.1rem',
                            color:
                              result.risk_category === 'Elevated'
                                ? '#ef4444'
                                : result.risk_category === 'Borderline'
                                ? '#f59e0b'
                                : '#10b981',
                          }}
                        >
                          {result.risk_category}
                        </Typography>
                      </Box>

                      <Box sx={{ p: 2, backgroundColor: 'rgba(255, 255, 255, 0.8)', borderRadius: 1.5, mt: 2, border: '1px solid rgba(0,0,0,0.05)' }}>
                        <Typography variant="body2" sx={{ fontWeight: 500, color: '#666', mb: 0.8 }}>
                          Threshold
                        </Typography>
                        <Typography variant="h6" sx={{ fontWeight: 700, fontSize: '1.1rem', color: '#0066cc' }}>
                          {result.threshold_mm.toFixed(2)} mm
                        </Typography>
                      </Box>
                    </Card>

                    {/* Clinical Interpretation */}
                    <Paper sx={{ p: 3, backgroundColor: '#fafafa', border: '1px solid #e8e8e8', borderRadius: 1.5 }}>
                      <Typography variant="h6" sx={{ fontWeight: 600, mb: 1.5, color: '#0066cc' }}>
                        Clinical Significance
                      </Typography>
                      <Typography variant="body2" sx={{ color: '#555', lineHeight: 1.8, fontSize: '0.95rem', mb: 2 }}>
                        {result.clinical_significance}
                      </Typography>
                      <ExportButtons
                        data={{
                          value_mm: result.value_mm,
                          threshold_mm: result.threshold_mm,
                          risk_category: result.risk_category,
                          clinical_significance: result.clinical_significance,
                          timestamp: new Date().toISOString(),
                        }}
                        filename="cimt_analysis"
                      />
                    </Paper>
                  </Box>
                </Grow>
              )}
            </Paper>
          </Grid>
        </Grid>
      </Box>
    </Box>
  );
};

export default CIMTPage;
