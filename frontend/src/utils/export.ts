/**
 * Utility functions for exporting analysis results
 */

export const exportAsJSON = (data: any, filename: string) => {
  const jsonString = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonString], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${filename}.json`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

export const exportAsCSV = (data: Record<string, any>, filename: string) => {
  const headers = Object.keys(data);
  const values = headers.map(h => {
    const val = data[h];
    if (typeof val === 'object') {
      return JSON.stringify(val);
    }
    return val;
  });

  const csvContent = [
    headers.join(','),
    values.join(','),
  ].join('\n');

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${filename}.csv`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

export const getTimestamp = () => {
  const now = new Date();
  return now.toISOString().slice(0, 19).replace(/:/g, '-');
};
