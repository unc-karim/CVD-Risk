/**
 * Fusion page - Full CVD risk assessment
 * MAIN PAGE - Combines all three models
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  Card,
  CircularProgress,
  Container,
  Grid,
  LinearProgress,
  Paper,
  TextField,
  Typography,
  Alert,
  
  Radio,
  RadioGroup,
  FormControlLabel,
  Grow,
} from '@mui/material';
import ImageUpload from '../components/ImageUpload';
import ExportButtons from '../components/ExportButtons';
import api, { FusionPrediction } from '../services/api';

interface FusionPageProps {
  apiReady: boolean;
}

const FusionPage: React.FC<FusionPageProps> = ({ apiReady }) => {
  const navigate = useNavigate();
  const [leftImage, setLeftImage] = useState<File | null>(null);
  const [rightImage, setRightImage] = useState<File | null>(null);
  const [age, setAge] = useState(65);
  const [gender, setGender] = useState(1);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<FusionPrediction | null>(null);
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

    if (age < 1 || age > 150) {
      setError('Please enter valid age (1-150)');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await api.predictFusion(leftImage, rightImage, age, gender);
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

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'High':
        return '#f44336';
      case 'Medium':
        return '#ff9800';
      case 'Low':
        return '#4caf50';
      default:
        return '#1976d2';
    }
  };

  return (
    <Box sx={{ background: '#ffffff', minHeight: '100vh' }}>
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
        <Container maxWidth="lg">
          <Box sx={{ display: 'flex', flexDirection: { xs: 'column', sm: 'row' }, alignItems: { xs: 'flex-start', sm: 'center' }, justifyContent: 'space-between', gap: { xs: 3, sm: 2, md: 4 } }}>
            <Box sx={{ flex: 1 }}>
              <Typography
                component="h1"
                variant="h2"
                sx={{
                  mb: 1,
                }}
              >
                Full CVD Risk Assessment
              </Typography>
              <Typography variant="body1" sx={{ opacity: 0.85 }}>
                Comprehensive cardiovascular risk analysis combining all models
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
        </Container>
      </Box>

      <Container maxWidth="lg" sx={{ pb: 6 }}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Paper sx={{ p: 3, backgroundColor: '#f8faff', maxWidth: 700, mx: 'auto' }}>
              <Typography variant="h6" sx={{ mb: 3, fontWeight: 600, color: '#0066cc' }}>
                Patient Information & Images
              </Typography>

              {/* Clinical Data Section - FIRST */}
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
                  disabled={loading}
                  sx={{ mb: 2 }}
                  variant="outlined"
                  size="small"
                />

                <Box sx={{ mb: 0 }}>
                  <Typography variant="body2" sx={{ mb: 1, fontWeight: 500, color: '#555' }}>
                    Sex
                  </Typography>
                  <RadioGroup value={gender.toString()} onChange={(e) => setGender(parseInt(e.target.value))} row>
                    <FormControlLabel value="0" control={<Radio size="small" disabled={loading} />} label="Female" />
                    <FormControlLabel value="1" control={<Radio size="small" disabled={loading} />} label="Male" />
                  </RadioGroup>
                </Box>
              </Box>

              {/* Retinal Images Section - now full width */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 600, color: '#555' }}>
                  Fundus Images
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <Box sx={{ width: '100%' }}>
                      <ImageUpload
                        onImageSelect={(file) => setLeftImage(file)}
                        label="Left Eye Image (Required)"
                        imageUrl={leftImage ? URL.createObjectURL(leftImage) : ''}
                        disabled={loading}
                      />
                      {leftImage && (
                        <Button
                          size="small"
                          variant="outlined"
                          color="error"
                          onClick={handleClearLeftImage}
                          sx={{ mt: 1, width: '100%' }}
                          disabled={loading}
                        >
                          Clear Image
                        </Button>
                      )}
                    </Box>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Box sx={{ width: '100%' }}>
                      <ImageUpload
                        onImageSelect={(file) => setRightImage(file)}
                        label="Right Eye Image (Optional)"
                        imageUrl={rightImage ? URL.createObjectURL(rightImage) : ''}
                        disabled={loading}
                      />
                      {rightImage && (
                        <Button
                          size="small"
                          variant="outlined"
                          color="error"
                          onClick={handleClearRightImage}
                          sx={{ mt: 1, width: '100%' }}
                          disabled={loading}
                        >
                          Clear Image
                        </Button>
                      )}
                    </Box>
                  </Grid>
                </Grid>
              </Box>

              {/* Action Button */}
              <Button
                variant="contained"
                fullWidth
                onClick={handlePredict}
                disabled={!leftImage || loading || !apiReady}
                size="large"
                sx={{ mt: 3, fontWeight: 700, fontSize: '1.1rem', py: 2 }}
              >
                {loading ? (
                  <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center' }}>
                    <CircularProgress size={20} color="inherit" />
                    Generating Analysis...
                  </Box>
                ) : (
                  '📋 Generate CVD Risk Assessment'
                )}
              </Button>

              {error && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  {error}
                </Alert>
              )}

              {/* Results Section - now below input */}
              {result && (
                <Grow in={true} timeout={500}>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3, mt: 4 }}>
                    {/* Step 1: HTN Model Results */}
                    <Paper sx={{ p: 3, backgroundColor: '#ffeaea', border: '2px solid #ff6f61', borderRadius: 2 }}>
                      <Typography variant="h6" sx={{ mb: 2.5, fontWeight: 700, color: '#c62828' }}>
                        1️⃣ Hypertension Detection (HTN Model)
                      </Typography>
                      <Grid container spacing={2}>
                        <Grid item xs={12} sm={6}>
                          <Paper sx={{ p: 2, backgroundColor: '#ffffff', borderRadius: 1 }}>
                            <Typography variant="body2" sx={{ color: '#666', mb: 0.5 }}>Status</Typography>
                            <Typography variant="h6" sx={{ fontWeight: 700, color: '#ff6f61' }}>
                              {result.hypertension.label}
                            </Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={12} sm={6}>
                          <Paper sx={{ p: 2, backgroundColor: '#ffffff', borderRadius: 1 }}>
                            <Typography variant="body2" sx={{ color: '#666', mb: 0.5 }}>Probability</Typography>
                            <Typography variant="h6" sx={{ fontWeight: 700, color: '#ff6f61' }}>
                              {(result.hypertension.probability * 100).toFixed(1)}%
                            </Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={12}>
                          <Typography variant="body2" sx={{ color: '#666' }}>
                            Confidence: <strong>{result.hypertension.confidence}</strong>
                          </Typography>
                        </Grid>
                      </Grid>
                    </Paper>

                    {/* Step 2: CIMT Model Results */}
                    <Paper sx={{ p: 3, backgroundColor: '#e3f2fd', border: '2px solid #64b5f6', borderRadius: 2 }}>
                      <Typography variant="h6" sx={{ mb: 2.5, fontWeight: 700, color: '#1565c0' }}>
                        2️⃣ CIMT Regression (Carotid Thickness Model)
                      </Typography>
                      <Grid container spacing={2}>
                        <Grid item xs={12} sm={6}>
                          <Paper sx={{ p: 2, backgroundColor: '#ffffff', borderRadius: 1 }}>
                            <Typography variant="body2" sx={{ color: '#666', mb: 0.5 }}>Thickness</Typography>
                            <Typography variant="h6" sx={{ fontWeight: 700, color: '#64b5f6' }}>
                              {result.cimt.value_mm.toFixed(2)} mm
                            </Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={12} sm={6}>
                          <Paper sx={{ p: 2, backgroundColor: '#ffffff', borderRadius: 1 }}>
                            <Typography variant="body2" sx={{ color: '#666', mb: 0.5 }}>Risk Category</Typography>
                            <Typography variant="h6" sx={{ fontWeight: 700, color: '#64b5f6' }}>
                              {result.cimt.risk_category}
                            </Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={12}>
                          <Typography variant="body2" sx={{ color: '#666' }}>
                            Threshold: {result.cimt.threshold_mm.toFixed(2)} mm | {result.cimt.clinical_significance}
                          </Typography>
                        </Grid>
                      </Grid>
                    </Paper>

                    {/* Step 3: Vessel Segmentation Results */}
                    <Paper sx={{ p: 3, backgroundColor: '#e8f5e9', border: '2px solid #81c784', borderRadius: 2 }}>
                      <Typography variant="h6" sx={{ mb: 2.5, fontWeight: 700, color: '#2e7d32' }}>
                        3️⃣ Vessel Segmentation (A/V Analysis Model)
                      </Typography>

                      {/* Segmentation Mask Visualization */}
                      {result.vessel.segmentation_mask_base64 && (
                        <Box sx={{ mb: 3 }}>
                          <Typography variant="body2" sx={{ color: '#666', mb: 1.5, fontWeight: 500 }}>Vessel Segmentation Mask</Typography>
                          <Card sx={{ overflow: 'hidden', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
                            <img
                              src={result.vessel.segmentation_mask_base64}
                              alt="Vessel Segmentation Mask"
                              style={{ width: '100%', display: 'block', borderRadius: 4 }}
                            />
                          </Card>
                        </Box>
                      )}

                      <Grid container spacing={2}>
                        <Grid item xs={12} sm={6}>
                          <Paper sx={{ p: 2, backgroundColor: '#ffffff', borderRadius: 1 }}>
                            <Typography variant="body2" sx={{ color: '#666', mb: 0.5 }}>Vessel Density</Typography>
                            <Typography variant="h6" sx={{ fontWeight: 700, color: '#81c784' }}>
                              {result.vessel.vessel_density.toFixed(3)}
                            </Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={12} sm={6}>
                          <Paper sx={{ p: 2, backgroundColor: '#ffffff', borderRadius: 1 }}>
                            <Typography variant="body2" sx={{ color: '#666', mb: 0.5 }}>Peripheral Density</Typography>
                            <Typography variant="h6" sx={{ fontWeight: 700, color: '#81c784' }}>
                              {result.vessel.features.peripheral_density.toFixed(3)}
                            </Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={12} sm={6}>
                          <Paper sx={{ p: 2, backgroundColor: '#ffffff', borderRadius: 1 }}>
                            <Typography variant="body2" sx={{ color: '#666', mb: 0.5 }}>Avg Vessel Width</Typography>
                            <Typography variant="h6" sx={{ fontWeight: 700, color: '#81c784' }}>
                              {result.vessel.features.avg_vessel_width.toFixed(2)}
                            </Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={12} sm={6}>
                          <Paper sx={{ p: 2, backgroundColor: '#ffffff', borderRadius: 1 }}>
                            <Typography variant="body2" sx={{ color: '#666', mb: 0.5 }}>Fractal Dimension</Typography>
                            <Typography variant="h6" sx={{ fontWeight: 700, color: '#81c784' }}>
                              {result.vessel.features.fractal_dimension.toFixed(3)}
                            </Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={12} sm={6}>
                          <Paper sx={{ p: 2, backgroundColor: '#ffffff', borderRadius: 1 }}>
                            <Typography variant="body2" sx={{ color: '#666', mb: 0.5 }}>Branching Density</Typography>
                            <Typography variant="h6" sx={{ fontWeight: 700, color: '#81c784' }}>
                              {result.vessel.features.branching_density.toFixed(4)}
                            </Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={12} sm={6}>
                          <Paper sx={{ p: 2, backgroundColor: '#ffffff', borderRadius: 1 }}>
                            <Typography variant="body2" sx={{ color: '#666', mb: 0.5 }}>Avg Tortuosity</Typography>
                            <Typography variant="h6" sx={{ fontWeight: 700, color: '#81c784' }}>
                              {result.vessel.features.avg_tortuosity.toFixed(3)}
                            </Typography>
                          </Paper>
                        </Grid>
                      </Grid>
                    </Paper>

                    {/* Step 4: Contributing Factors */}
                    <Paper sx={{ p: 3, backgroundColor: '#f3e5f5', border: '2px solid #b39ddb', borderRadius: 2 }}>
                      <Typography variant="h6" sx={{ mb: 2.5, fontWeight: 700, color: '#6a1b9a' }}>
                        4️⃣ Contributing Factors Analysis
                      </Typography>
                      <Grid container spacing={2}>
                        <Grid item xs={12} sm={6} md={4}>
                          <Paper sx={{ p: 2, backgroundColor: '#ffffff', borderRadius: 1 }}>
                            <Typography variant="body2" sx={{ color: '#666', mb: 0.5 }}>Hypertension Risk</Typography>
                            <LinearProgress
                              variant="determinate"
                              value={result.contributing_factors.hypertension_probability * 100}
                              sx={{ mb: 1, height: 6, borderRadius: 3 }}
                            />
                            <Typography variant="h6" sx={{ fontWeight: 700, color: '#b39ddb' }}>
                              {(result.contributing_factors.hypertension_probability * 100).toFixed(1)}%
                            </Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={12} sm={6} md={4}>
                          <Paper sx={{ p: 2, backgroundColor: '#ffffff', borderRadius: 1 }}>
                            <Typography variant="body2" sx={{ color: '#666', mb: 0.5 }}>CIMT Elevated</Typography>
                            <Typography variant="h6" sx={{ fontWeight: 700, color: result.contributing_factors.cimt_elevated ? '#d32f2f' : '#2e7d32' }}>
                              {result.contributing_factors.cimt_elevated ? '⚠️ Yes' : '✓ No'}
                            </Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={12} sm={6} md={4}>
                          <Paper sx={{ p: 2, backgroundColor: '#ffffff', borderRadius: 1 }}>
                            <Typography variant="body2" sx={{ color: '#666', mb: 0.5 }}>Vessel Abnormalities</Typography>
                            <LinearProgress
                              variant="determinate"
                              value={Math.min(result.contributing_factors.vessel_abnormalities * 10, 100)}
                              sx={{ mb: 1, height: 6, borderRadius: 3 }}
                            />
                            <Typography variant="h6" sx={{ fontWeight: 700, color: '#b39ddb' }}>
                              {result.contributing_factors.vessel_abnormalities.toFixed(2)}
                            </Typography>
                          </Paper>
                        </Grid>
                      </Grid>
                    </Paper>

                    {/* Step 5: Final CVD Risk Assessment */}
                    <Card
                      sx={{
                        backgroundColor: getRiskColor(result.risk_level),
                        color: 'white',
                        p: 4,
                        borderRadius: 2,
                        boxShadow: '0 12px 32px rgba(0,0,0,0.25)',
                      }}
                    >
                      <Typography variant="overline" sx={{ mb: 1, opacity: 0.9, fontWeight: 600, letterSpacing: 1.5, fontSize: '0.85rem' }}>
                        5️⃣ Final Assessment
                      </Typography>
                      <Typography variant="h2" sx={{ fontWeight: 800, mb: 3, fontSize: '3.2rem', lineHeight: 1.1 }}>
                        {result.risk_level} Risk
                      </Typography>

                      <Box sx={{ mb: 3 }}>
                        <Typography variant="body1" sx={{ mb: 1.5, fontWeight: 600, fontSize: '1rem', opacity: 0.95 }}>
                          CVD Risk Probability
                        </Typography>
                        <Typography variant="h3" sx={{ fontWeight: 800, mb: 2, fontSize: '2.4rem' }}>
                          {(result.cvd_probability * 100).toFixed(1)}%
                        </Typography>

                        <LinearProgress
                          variant="determinate"
                          value={result.cvd_probability * 100}
                          sx={{
                            height: 12,
                            borderRadius: 3,
                            backgroundColor: 'rgba(255, 255, 255, 0.3)',
                            '& .MuiLinearProgress-bar': {
                              backgroundColor: 'white',
                              boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
                            },
                          }}
                        />
                      </Box>
                    </Card>

                    {/* Recommendation */}
                    <Paper sx={{ p: 3, backgroundColor: '#fff3e0', border: '2px solid #ff9800', borderRadius: 2 }}>
                      <Typography variant="h6" sx={{ mb: 2, fontWeight: 700, color: '#e65100' }}>
                        💡 Recommendation
                      </Typography>
                      <Typography variant="body2" sx={{ color: '#555', lineHeight: 1.8 }}>
                        {result.recommendation}
                      </Typography>
                      <Box sx={{ mt: 2 }}>
                        <Typography variant="caption" sx={{ color: '#999', display: 'block' }}>
                          Processing time: {result.processing_time_seconds.toFixed(2)}s
                        </Typography>
                      </Box>
                    </Paper>

                    {/* Export */}
                    <Paper sx={{ p: 3, backgroundColor: '#f5f5f5', borderRadius: 2 }}>
                      <Typography variant="body2" sx={{ mb: 2, fontWeight: 600, color: '#555' }}>
                        Export Full Analysis
                      </Typography>
                      <ExportButtons
                        data={{
                          patient_info: { age, gender: gender === 0 ? 'Female' : 'Male' },
                          htn_result: result.hypertension,
                          cimt_result: result.cimt,
                          vessel_result: {
                            vessel_density: result.vessel.vessel_density,
                            features: result.vessel.features,
                          },
                          contributing_factors: result.contributing_factors,
                          cvd_assessment: {
                            risk_level: result.risk_level,
                            cvd_probability: result.cvd_probability,
                            recommendation: result.recommendation,
                          },
                          timestamp: new Date().toISOString(),
                        }}
                        filename="fusion_cvd_assessment"
                      />
                    </Paper>
                  </Box>
                </Grow>
              )}
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default FusionPage;
