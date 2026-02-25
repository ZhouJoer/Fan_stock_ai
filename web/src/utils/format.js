/**
 * 日期时间格式化
 * @param {string} iso - ISO 日期字符串
 * @returns {string}
 */
export function formatTs(iso) {
    try {
        return new Date(iso).toLocaleString('zh-CN')
    } catch {
        return iso
    }
}
