/**
 * src/interfaces/frontend/src/utils/formatters.ts
 * Các hàm định dạng thời gian, kích thước tệp, regex và màu avatar,
 * tương thích với renderer.js và app.js.
 */

export function formatTime(timestamp?: number | Date | null): string {
  if (!timestamp) return "";
  const d = timestamp instanceof Date ? timestamp : new Date(timestamp);
  if (Number.isNaN(d.getTime())) return "";
  const hours = String(d.getHours()).padStart(2, "0");
  const minutes = String(d.getMinutes()).padStart(2, "0");
  return `${hours}:${minutes}`;
}

export function formatDateHeader(timestamp: number | Date): string {
  const d = timestamp instanceof Date ? timestamp : new Date(timestamp);
  if (Number.isNaN(d.getTime())) return "";

  const today = new Date();
  if (d.toDateString() === today.toDateString()) {
    return "HÔM NAY";
  }

  const yesterday = new Date(today);
  yesterday.setDate(today.getDate() - 1);
  if (d.toDateString() === yesterday.toDateString()) {
    return "HÔM QUA";
  }

  return `${d.getDate()}/${d.getMonth() + 1}/${d.getFullYear()}`;
}

export function formatFileSize(bytes?: number | null): string {
  if (!bytes || bytes <= 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.min(
    sizes.length - 1,
    Math.floor(Math.log(bytes) / Math.log(k))
  );
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

export function escapeRegex(value?: string | null): string {
  if (!value) return "";
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function hashCode(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i += 1) {
    hash = (hash << 5) - hash + str.charCodeAt(i);
    hash |= 0;
  }
  return hash;
}

const NO_ANSWER_PATTERNS: readonly RegExp[] = [
  /chưa có thông tin cụ thể/i,
  /không có thông tin cụ thể/i,
  /không có thông tin về/i,
  /không tìm thấy thông tin/i,
  /chưa tìm thấy thông tin/i,
  /không có dữ liệu về/i,
  /chưa có dữ liệu về/i,
  /tài liệu (?:quy chế )?(?:lưu trữ )?chưa có thông tin/i,
  /ngữ cảnh (?:được cung cấp )?(?:hiện )?không có thông tin/i,
  /không đủ (?:cơ sở|dữ liệu|thông tin) để (?:trả lời|khẳng định)/i,
  /ngoài phạm vi/i,
  /không thuộc phạm vi/i,
  /không nằm trong phạm vi/i,
  /chưa được đề cập/i,
  /không được đề cập/i,
  /không phải là thông tin thuộc/i,
  /không liên quan/i,
  /nội dung (?:em|bạn)?(?: vừa)?(?: chia sẻ| hỏi)? không liên quan/i,
  /xin lỗi[\s\S]{0,60}văn bản quy chế[\s\S]{0,60}chưa có/i,
  /vui lòng liên hệ trực tiếp phòng đào tạo/i,
  /mất nhiều thời gian xử lý hơn dự kiến/i,
  /tạm thời quá tải/i,
  /tạm thời bị gián đoạn/i,
  /gặp sự cố kỹ thuật tạm thời/i,
  /vượt quá giới hạn xử lý/i,
];

export function isNoAnswerOrOutOfDomain(text?: string | null): boolean {
  const s = (text || "").trim();
  if (!s) return false;
  return NO_ANSWER_PATTERNS.some((pattern) => pattern.test(s));
}

export function stripMarkdownForSearch(raw?: string | null): string {
  if (!raw) return "";
  return raw
    .replace(/[*_~`#>]+/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

